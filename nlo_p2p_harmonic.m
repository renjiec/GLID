function [phipsyIters, allStats] = nlo_p2p_harmonic(invM_AQP, D2, C2, bP2P, softP2P, lambda, phipsyIters, energy_parameter, AQPKappa, nIter, solver, energy_type, ...
    nextSampleInSameCage, hessianSampleRate, fillDistanceSegments, v, E2, L)

linesearchLIM = nargin>12;
linesearchLIM2 = true;

if numel(v)==1, v = v{1}; end

% energy_types = {'ARAP', 'ISO', 'EISO', 'AMIPS'};

n = size(D2,2);
numEnergySamples = size(D2, 1);

enEvalsPerKernel = 10;

hessSampleStride = ceil(1/hessianSampleRate);
hessian_samples = 1: hessSampleStride :ceil(numEnergySamples-hessSampleStride/2);
if hasGPUComputing, hessian_samples = gpuArray(hessian_samples); end
hessianSampleRate = numel(hessian_samples)/numEnergySamples;

optimization_methods = {'GD', 'AQP', 'Newton', 'Newton_SPDH', 'Newton_SPDH_FullEig'};
CUDA_SOLVER_NAMES = {'cuGD', 'cuNewton', 'cuNewton_SPDH', 'cuNewton_SPDH_FullEig'};
switch solver
    case CUDA_SOLVER_NAMES
        assert( ~iscell(v), 'multi-connected domain is not yet supported in CUDA implementation');
        singlePrecision = numel(solver)>7 && strcmpi(solver(end-6:end), ' single');
        if singlePrecision
            solver = solver(1:end-7); 
            fillDistanceSegments = single(fillDistanceSegments);
            v = single(v); L = single(L); E2 = single(E2);
        end
            
        optmethod = find(strcmpi(solver(3:end), optimization_methods), 1) - 1;

        params = struct('hessian_samples', int32(hessian_samples-1), 'isometric_energy_power', energy_parameter, 'aqp_kappa', AQPKappa, 'nIter', nIter,  ...
                        'sample_spacings_half', fillDistanceSegments, 'v', v, 'E2', E2, 'L', L, ...
                        'LS_energy_eval_per_kernel', enEvalsPerKernel, 'solver', optmethod, 'linearSolvePref', 0, 'deltaFixSPDH', 1e-15, 'reportIterationStats', 1);

        [phipsyIters, allStats] = cuHarmonic( invM_AQP, D2, C2, bP2P, lambda, double(phipsyIters), params);

        if size(allStats,1)>2
            allStats = allStats';
        else
            allStats = [zeros(1, 7) allStats];
        end
        return;
end

epow = energy_parameter;

fDistortionGradImp = @(fzalpha1, gzalpha2, evec) 2*[D2'*(fzalpha1.*evec); conj(D2'*(gzalpha2.*evec))]; 

%% energy functions for different energy types
switch energy_type
    case 'ARAP'
        fGradIso = @(fzgzb, fzgzb2) 4*[D2'*(fzgzb(:,1).*(1-1./abs(fzgzb(:,1)))); conj(D2'*fzgzb(:,2))]; 
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( 2*sum(fzgz2, 2)-4*fzgz2(:,1).^0.5 + 2 );
        linesearchLIM = false;
        linesearchLIM2 = false;
    case 'ISO'
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( 2*sum(fzgz2, 2).*(1+diff(fzgz2, 1, 2).^-2) );
        fGradIso = @(fzgz, fzgz2) 4*[D2'*(fzgz(:,1).*(1-((fzgz2*[1;-1]).^-3).*(fzgz2*[1;3])) ); conj(D2'*(fzgz(:,2).*(1+((fzgz2*[1;-1]).^-3).*(fzgz2*[3;1]))))];
    case 'EISO'
        fIsometryicEnergyFzGz2 = @(fzgz2) sum( exp(sum(fzgz2, 2).*(1+diff(fzgz2, 1, 2).^-2)) );
        fGradIso = @(fzgz, fzgz2) fDistortionGradImp( fzgz(:,1).*(1+(diff(fzgz2,1,2).^-3).*(fzgz2*[1;3])), ...
                                                      fzgz(:,2).*(1-(diff(fzgz2,1,2).^-3).*(fzgz2*[3;1])), ...
                                                      exp( sum(fzgz2,2).*(1+diff(fzgz2,1,2).^-2) ) );
    case 'AMIPS'
        mu = energy_parameter;
        fIsometryicEnergyFzGz2 = @(fzgz2) -sum( (mu*2*sum(fzgz2,2)+1)./diff(fzgz2,1,2) + diff(fzgz2,1,2) );
        fGradIso = @(fzgz, fzgz2) 2*[D2'*( fzgz(:,1).*(1-(4*mu*fzgz2(:,2)+1).*diff(fzgz2,1,2).^-2)); 
                                conj(D2'*(-fzgz(:,2).*(1-(4*mu*fzgz2(:,1)+1).*diff(fzgz2,1,2).^-2)))];                             
end


fP2PEnergyPhiPsy = @(fg) norm(fg(:,1)+conj(fg(:,2)) - bP2P)^2;

matGradP2P = 2*[C2 conj(C2)]'*[C2 conj(C2) -bP2P];
fGradP2P = @(phi, psy) matGradP2P*[phi; conj(psy); ones(1,size(phi,2))];



fGradIso_LBFGS = @(fzgz, fzgz2) 4*[D2'*arrayfun(@(x, y, z)  z*(1-(x-y)^-3*(x+3*y)), fzgz2(:,1:2:end), fzgz2(:,2:2:end), fzgz(:,1:2:end)); ...
                              conj(D2'*arrayfun(@(x, y, z)  z*(1+(x-y)^-3*(3*x+y)), fzgz2(:,1:2:end), fzgz2(:,2:2:end), fzgz(:,2:2:end)))];

fC2Rm = @(x) [real(x) -imag(x); imag(x) real(x)];
fR2Cv = @(x) complex(x(1:end/2), x(end/2+1:end));
fC2Rv = @(x) [real(x); imag(x)];

CtC = [C2 conj(C2)]'*[C2 conj(C2)];
CtCr = fC2Rm(CtC);
CtCr(1:n*3, n*3+(1:n)) = -CtCr(1:n*3, n*3+(1:n));
CtCr(n*3+(1:n), 1:n*3) = -CtCr(n*3+(1:n), 1:n*3);

ls_beta = 0.5;
ls_alpha = 0.2;

fMyEnergy = @(fzgz2, fg) fIsometryicEnergyFzGz2(fzgz2) + fP2PEnergyPhiPsy(fg)*lambda;
fMyGrad = @(fzgz, fzgz2, phi, psy) fGradIso(fzgz, fzgz2) + fGradP2P(phi, psy)*lambda;

%% nullspace, reduce problem size by removing redundant vars
if ~iscell(v)
    N = speye(2*n, 2*n-1);
    nHoles = 0;
else
    cageSz = numel(v{1});
    isFreePhi = true(1, n);
    isFreeVar = [isFreePhi isFreePhi];
    nHoles = numel(v)-1;
    isFreeVar([n+cageSz 2*n+(1-nHoles:0)]) = false;
    N = sparse([find(isFreeVar) 2*n+(1-nHoles:0)], [1:sum(isFreeVar) n+(1-nHoles:0)],  1, 2*n, sum(isFreeVar));
end

if hasGPUComputing && verLessThan('matlab', '9.2'), N = full(N); end

%%
if strcmpi(solver, 'Newton_ConformalMap')
    N = N(:, 1:n-nHoles);
end

% for real variables
Nr = blkdiag(N, N);
Nr(end-numel(v)+1:end,:) = -Nr(end-numel(v)+1:end,:);

switch solver
    case CUDA_SOLVER_NAMES % processed before, pass
    
    case 'LBFGS'
        %% using Dirichlet energy hessian as initial hessian, already reordered by real/imag of phi psy
%         h = 2*blkdiag(D2'*D2, conj(D2'*D2));

        %% identity initial hessian for isometric energy
        h = eye(2*n)*2;
        M = h + 2*lambda*CtC;

        %% for LBFGS, a PSD initial hessian is need for the algorithm to converge, so remove the 
        % global DOF of harmonic map from the problem
        invM = N*inv(N'*M*N)*N';

        LBFGS_K = 5; % parameter k

        fzgzAll = D2*phipsyIters;
        gIterAll = fGradIso_LBFGS(fzgzAll, abs(fzgzAll).^2) + fGradP2P(phipsyIters(:, 1:2:end), phipsyIters(:, 2:2:end))*lambda;
        
        allStats = zeros(nIter+1, 8);
        allStats(1, 8) = gather( fMyEnergy( abs(fzgzAll(:,1:2)).^2, C2*phipsyIters(:,1:2)) );
        for it=1:nIter
            tic

            %% iteration
            fzgz = fzgzAll(:, 1:2);
            ppIter = reshape(phipsyIters, n*2, []);
            ppIter(n+1:end,:) = conj( ppIter(n+1:end,:) );
            fgP2P = C2*phipsyIters(:, 1:2);
            
            dpp = lbfgs_iter(invM, gIterAll, ppIter);
            dppdotg = dot( [real(dpp); imag(dpp)], [real(gIterAll(:,1)); imag(gIterAll(:,1))] );
            dpp = reshape(dpp, [], 2);
            
            normdpp = norm(dpp);            
            dpp(:,2) = conj(dpp(:,2));
            
            dfzgz = D2*dpp;
            dfgP2P = C2*dpp;
            fMyFun = @(t) fMyEnergy(abs(fzgz+t*dfzgz).^2, fgP2P+t*dfgP2P);

            maxts = arrayfun(@maxtForPhiPsy, fzgz(:,1), fzgz(:,2), dfzgz(:,1), dfzgz(:,2));
            ls_t = min(1, min(maxts)*0.8); % faster than min( [maxts;1] )
            
            e = fMyEnergy(abs(fzgz).^2, fgP2P);
            fQPEstim = @(t) e+ls_alpha*t*dppdotg;
            e_new = fMyFun(ls_t);
            while ls_t*normdpp>1e-12 && e_new > fQPEstim(ls_t)
                ls_t = ls_t/2;
                e_new = fMyFun(ls_t);
            end

            if linesearchLIM
                ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsyIters(:,1:2), dpp, fzgz(:,1:2), dfzgz(:,1:2), ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage);
                e_new = fMyFun(ls_t);
            end

            phipsyIters = [phipsyIters(:,1:2)+ls_t*dpp phipsyIters(:,1:min(end,2*(LBFGS_K-1)))];

            %% update for next iteration
            fzgz = D2*phipsyIters(:,1:2);

            gIter = fGradIso_LBFGS(fzgz, abs(fzgz).^2) + fGradP2P(phipsyIters(:, 1), phipsyIters(:, 2))*lambda;
            gIterAll = [gIter gIterAll(:, 1:min(end, (LBFGS_K-1)))];
            fzgzAll = [fzgz fzgzAll(:, 1:min(end, 2*(LBFGS_K-1)))];

            allStats(it+1, [5 8]) = [toc*1000 gather( real(e_new))];
            fprintf('LBFGS it %d: runtime: %.3es, energy: %.3e\n', it, allStats(it,5)/1000, allStats(it,8));
        end

    case {'Newton', 'Newton_SPDH', 'Newton_ConformalMap', 'Newton_SPDH_FullEig', 'Gradient Descent'}
        phipsyIters = double(phipsyIters);

        phi = phipsyIters(:, 1);
        psy = phipsyIters(:, 2);
        
        fzgz0 = D2*[phi psy];   % gz: conj(fzb)
        fzgz2 = abs(fzgz0).^2;  % abs2([fz conj(fzb)])
        
        e = fMyEnergy(fzgz2, C2*[phi psy]);
        
        % statistics to be in consistant with that returned from cuda solvers
        allStats = zeros(nIter+1, 8);
        allStats(1, [7 8]) = gather( [fP2PEnergyPhiPsy(C2*[phi psy])*lambda e] );

        for it=1:nIter
            tic;
            fgP2P = C2*[phi psy];
            g = fMyGrad(fzgz0, fzgz2, phi, psy);

            if strncmpi(solver, 'Newton', 6)
                SPDHessian = strcmp(solver, 'Newton_SPDH');
                
                switch energy_type
                case 'ARAP'
                    [~, ~, h] = harmonicMapARAPEnergyReal(D2(hessian_samples,:), phi, psy, SPDHessian);
                case 'ISO'
                    [~, ~, h] = harmonicMapIsometryicEnergyReal(D2(hessian_samples,:), phi, psy, epow, SPDHessian);
                case 'EISO'
                    [~, ~, h] = harmonicMapExpIsometryicEnergy(D2(hessian_samples,:), phi, psy, energy_parameter, SPDHessian);
                case 'AMIPS'
                    [~, ~, h] = harmonicMapAMIPSEnergyReal(D2(hessian_samples,:), phi, psy, mu, SPDHessian);
                end

                h = h/hessianSampleRate;

                if strcmpi(solver, 'Newton_SPDH_FullEig')
                    [eigD, eigE] = eig( tril(h) + tril(h)' - diag(diag(h)) );
                    eigE(eigE<0) = 0;
                    h = eigD*eigE*eigD';
                end

                M = h + 2*lambda*CtCr;
            end

            %% solve
            if strcmp(solver, 'Gradient Descent')
                dpp = -g;
            elseif strncmpi(solver, 'Newton', 6)
                g(n+1:n*2) = conj( g(n+1:n*2) );
                dpp = fR2Cv( Nr*( (Nr'*M*Nr)\(Nr'*fC2Rv(-g)) ) );
            end

            dppdotg = dot( [real(dpp); imag(dpp)], [real(g); imag(g)] );
            normdpp = norm(dpp);

            if ~strncmpi(solver, 'Newton', 6)  % no conjugate is needed for newton because of using real numbers
                dpp = [dpp(1:n); conj( dpp(n+1:2*n) )];
            end

            dfzgz = D2*reshape(dpp, [], 2);
            dfgP2P = C2*reshape(dpp, [], 2);

            %%
            ls_t = 1;
            if linesearchLIM2
                maxts = arrayfun(@maxtForPhiPsy, fzgz0(:,1), fzgz0(:,2), dfzgz(:,1), dfzgz(:,2));
                ls_t = min(1, min(maxts)*0.8); % faster than min( [maxts;1] )
            end

            fQPEstim = @(t) e+ls_alpha*t*dppdotg;
            fMyFun = @(t) fMyEnergy(abs(fzgz0+t*dfzgz).^2, fgP2P+t*dfgP2P);

            e_new = fMyFun(ls_t);
            while ls_t*normdpp>1e-12 && e_new > fQPEstim(ls_t)
                ls_t = ls_t*ls_beta;
                e_new = fMyFun(ls_t);
            end
            e = e_new;
            
            dpp = reshape(dpp,[],2);
            
            if linesearchLIM
                ls_t = lineSearchLocallyInjectiveHarmonicMap(phipsyIters(:,1:2), dpp, fzgz0, dfzgz, ls_t, fillDistanceSegments, v, E2, L, nextSampleInSameCage);
                e = fMyFun(ls_t);
            end

            allStats(it+1, [5 7 8]) = gather( [toc*1000 fP2PEnergyPhiPsy(fgP2P+ls_t*dfgP2P)*lambda e] );

            if ls_t*normdpp<1e-12, break; end

            phi = phi + ls_t*dpp(:,1);
            psy = psy + ls_t*dpp(:,2);

            fzgz0 = fzgz0 + ls_t*dfzgz;
            fzgz2 = abs(fzgz0).^2;

            phipsyIters = [phi psy phipsyIters(:,1:end-2)];
        end

    otherwise
        warning('Unexpected solver type: %s. ', solver);
end
