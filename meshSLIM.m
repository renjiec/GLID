function [z, allStats] = meshSLIM(x, t, P2PVtxIds, P2PCurrentPositions, z, nIter, lambda, energy_type, energy_parameter)
% z: initilization

if isempty(P2PVtxIds), z=x; allStats=zeros(1,8); return; end

if nargin < 7, lambda = 1e9; end
if nargin < 8, energy_type = 'SymmDirichlet'; end
if nargin < 9, energy_parameter = 1; end

fC2R = @(x) [real(x) imag(x)];

nv = numel(x);

e1 = [2 3 1];   e2 = [3 1 2];
VE = full( sparse( [1:3 1:3], [e1 e2], -[1 1 1 -1 -1 -1]) );

e = x(t(:,e2)) - x(t(:,e1));

fComputeM = @(x) (x(:,[5 4 4 2 1 1])-x(:,[6 6 5 3 3 2]))./((dot(x(:,[1 4 2]),x(:,[5 3 6]),2)-dot(x(:,1:3), x(:,[6 4 5]),2))*[1 -1 1 -1 1 -1]);
M1 = fComputeM( fC2R(x(t)) );

fComputeJacobR = @(M, z) [M(:,[1 1]).*z(t(:,1), :)+M(:,[2 2]).*z(t(:,2), :)+M(:,[3 3]).*z(t(:,3), :) M(:,[4 4]).*z(t(:,1), :)+M(:,[5 5]).*z(t(:,2), :)+M(:,[6 6]).*z(t(:,3), :)];

%% functors
fNormalize = @(x) x./abs(x);
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

Areas = signedAreas(x, t);

P2Plhs = 2*lambda*sparse(P2PVtxIds, P2PVtxIds, 1, nv, nv);
P2Prhs = 2*lambda*sparse(P2PVtxIds, 1, P2PCurrentPositions, nv, 1);


switch energy_type
case 'ARAP'
    fIsoEnergy = @(sigs) dot( Areas, sum( (sigs-1).^2, 2) );
case 'SymmDirichlet'
    fIsoEnergy = @(sigs) dot( Areas, sum(sigs.^2+sigs.^-2, 2) );
case 'Exp_SymmDirichlet'
    fIsoEnergy = @(sigs) dot( Areas, exp( sum(sigs.^2+sigs.^-2, 2)*energy_parameter ) );
    otherwise
    warning('not supported energy: %s!', energy_type);
end


fComputeSigmas = @(z) abs( fComputeJacobR(M1, fC2R(z))*[1 1; -1i 1i; 1i 1i; 1 -1]/2 )*[1 1; 1 -1];
fDeformEnergy = @(z) fIsoEnergy( fComputeSigmas(z) ) + lambda*norm(z(P2PVtxIds)-P2PCurrentPositions)^2;

en = fDeformEnergy(z);
ls_beta = 0.5;
ls_alpha = 0.2;


allStats = zeros(nIter+1, 8); % statistics
allStats(1, [5 7 8]) = [0 norm(z(P2PVtxIds)-P2PCurrentPositions)^2 en];

for it=1:nIter
    tic
    
    %% local
    affs = fComputeJacobR(M1, fC2R(z));
    Rots = fBestRots(affs);

    %% compute differentials
    fz = affs*[1; -1i; 1i; 1]/2;
    gz = affs*[1; 1i; 1i; -1]/2;
    S = abs([fz gz])*[1 1; 1 -1];

    %% SLIM
    U2 = fz.*gz;
    U2(abs(U2)<1e-10) = 1;  % avoid division by 0
    U2 = U2./abs(U2);
    
    switch energy_type
    case 'ARAP'
        Sw = [1 1];
    case 'SymmDirichlet'
        Sw = (S.^-2+1).*(S.^-1+1);
    case 'Exp_SymmDirichlet'
        Sw = (S.^-2+1).*(S.^-1+1).*exp(energy_parameter*(S.^2+S.^-2))*2*energy_parameter;
    otherwise
        assert('energy %s not implemented for SLIM', energy_type);
    end
    
    
    Sv = Sw*[1 -1; 1 1]/2;  % sv1 > sv2
    
    e2 = Sv(:,1).*e + Sv(:,2).*U2.*conj(e);

    w2 = real( e(:,[2 3 1]).*conj(e2(:,[3 1 2])) )./Areas;
    L2 = sparse( t(:,[2 3 1 3 1 2]), t(:,[3 1 2 2 3 1]), [w2 w2]/2, nv, nv );
    L2 = spdiags(-sum(L2,2), 0, L2);

    %% global poisson
    b = accumarray( reshape(t,[],1), reshape(e2*1i.*Rots, [], 1) );
    
    z2 = (L2+P2Plhs) \ (b+P2Prhs);
    
    g = (L2+P2Plhs)*z - (b+P2Prhs);
    
    %% orientation preservation
    ls_t = min( min( maxtForPositiveArea( z(t)*VE(:,1:2), z2(t)*VE(:,1:2) ) )*0.9, 1 );
    
    %% line search energy decreasing
    fMyFun = @(t) fDeformEnergy( z2*t + z*(1-t) );
    normdz = norm(z-z2);
    dgdotfz = dot( [real(g); imag(g)], [real(z2-z); imag(z2-z)] );
    fQPEstim = @(t) en+ls_alpha*t*dgdotfz;

    e_new = fMyFun(ls_t);
    while ls_t*normdz>1e-12 && e_new > fQPEstim(ls_t)
        ls_t = ls_t*ls_beta;
        e_new = fMyFun(ls_t);
    end
    en = e_new;

    fprintf('it: %3d, t: %.3e, en: %.3e\n', it, ls_t, en);
    
    %% update
    z = z2*ls_t + z*(1-ls_t);
    
    %% stats
    allStats(it+1, [5 7 8]) = [toc*1000 norm(z(P2PVtxIds)-P2PCurrentPositions)^2 en];
end

allStats(:,7:8) = allStats(:,7:8)/sum(Areas);


fprintf('%dits: mean runtime: %.3e\n', nIter, mean(allStats(2:end,5)));
    
    