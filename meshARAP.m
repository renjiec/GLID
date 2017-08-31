function z = meshARAP(x, t, P2PVtxIds, P2PCurrentPositions, z, nIter, lambda)
% z: initilization

if nargin < 7, lambda = 1e9; end

fC2R = @(x) [real(x) imag(x)];

nv = numel(x);
e = x(t(:,[3 1 2])) - x(t(:,[2 3 1]));

fComputeM = @(x) (x(:,[5 4 4 2 1 1])-x(:,[6 6 5 3 3 2]))./((dot(x(:,[1 4 2]),x(:,[5 3 6]),2)-dot(x(:,1:3), x(:,[6 4 5]),2))*[1 -1 1 -1 1 -1]);
M1 = fComputeM( fC2R(x(t)) );

fComputeJacob = @(M, z) [M(:,[1 1]).*z(t(:,1), :)+M(:,[2 2]).*z(t(:,2), :)+M(:,[3 3]).*z(t(:,3), :) M(:,[4 4]).*z(t(:,1), :)+M(:,[5 5]).*z(t(:,2), :)+M(:,[6 6]).*z(t(:,3), :)];

fNormalize = @(x) x./abs(x);
fBestRots = @(a) fNormalize( complex(a(:,1)+a(:,4), a(:,2)-a(:,3)) );

P2Plhs = 2*lambda*sparse(P2PVtxIds, P2PVtxIds, 1, nv, nv);
P2Prhs = 2*lambda*sparse(P2PVtxIds, 1, P2PCurrentPositions, nv, 1);

L = cotLaplace(x, t);

for it=1:nIter
    %% local
    affs = fComputeJacob(M1, fC2R(z));
    R = fBestRots(affs);

    %% global poisson
    b = accumarray( reshape(t,[],1), reshape(e*1i.*R, [], 1) );
    z = (L+P2Plhs) \ (b+P2Prhs);
end
