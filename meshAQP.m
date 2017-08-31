function [z, allStats, meshXs] = meshAQP(x, t, P2PVtxIds, P2PCurrentPositions, z, nIter)

fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));


if ~exist('OptimProblemIsoDist','file')
    addpath( genpath([pwd '\meshAQP']) );
end


tic
if norm(z(P2PVtxIds) - P2PCurrentPositions)>1e-6
    initArapIter = 25; %1
    z = meshARAP(x, t, P2PVtxIds, P2PCurrentPositions, z, initArapIter);
end

nv = numel(x);
nP2P = numel(P2PVtxIds);
eq_lhs = sparse( 1:nP2P*2, [P2PVtxIds; P2PVtxIds+nv], 1, nP2P*2, nv*2 );
eq_rhs = [real(P2PCurrentPositions); imag(P2PCurrentPositions)];

optimProblem = OptimProblemIsoDist(fC2R(x), t, eq_lhs, eq_rhs, fC2R(z));
preprocessTime = toc;

%% setup solver
useAccelaration = true;
aqpSolver = OptimSolverAcclQuadProx('AQP', optimProblem, useAccelaration, true, true); aqpSolver.setKappa(1000);

%% solve
TolX = 1e-10;
TolFun = 1e-6;
logs = aqpSolver.solveTol(TolX, TolFun, nIter);

z = fR2C( logs.X(:, :, end) );

if nargout>2, meshXs = logs.X; end


allStats = zeros(nIter+1, 8); % statistics
allStats(:, 5) = [preprocessTime; logs.t_iter(2:end)']*1000; 
allStats(:, 8) = logs.f/sum(signedAreas(x,t));


% figuredocked; h = drawmesh(T, logs.X(:,:,end));
% 
% set(h, 'FaceColor', 'w', 'edgealpha', 0.05);
% for i=2:size(logs.X,3)
%     title( sprintf('iter %d', i) );
%     pause(0.2);
%     set(h, 'Vertices', logs.X(:,:,i));
% end


