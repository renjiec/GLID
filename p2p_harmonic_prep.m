%% set parameters

if exist('cage_offset', 'var')~=1
    cage_offset = 1e-1;
end

% cage_offset = 5e-2;

if exist('numVirtualVertices', 'var')~=1
    fprintf('setting default parameters for p2p-harmonic deformation\n');
    numVirtualVertices = 1;
    numFixedSamples = 1;
    numEnergySamples = 10000;
    numDenseEvaluationSamples = 15000;
    numActiveSetPoolSamples = 2000;
%     solver_type = 'Direct Mosek';

    no_output = true;
    p2p_weight = 100;
    sigma2_lower_bound = 0.35;
    sigma1_upper_bound = 5; 
    k_upper_bound = 0.8;
    binarySearchValidMap = true;
end

AQP_preprocessed = false;

if exist('softP2P', 'var')~=1,          softP2P = true;         end
if exist('AQP_kappa', 'var')~=1,        AQP_kappa = 1;          end
if hasGPUComputing
    AQP_nIterPerViz = 5;
else
    AQP_nIterPerViz = 1;
end
if exist('energy_parameter', 'var')~=1, energy_parameter = 1;   end
if exist('hessianSampleRate', 'var')~=1,hessianSampleRate = .1; end

if true %for deformation with isometric energy
    numDenseEvaluationSamples = numEnergySamples;
end


%%
offsetCage = subdivPolyMat(cage, numVirtualVertices-sum( cellfun(@numel, holes) )-1)*polygonOffset(cage, -cage_offset, false);
offsetHoles = cellfun(@(h) polygonOffset(h, -cage_offset, false), holes, 'UniformOutput', false);
holeCenters = reshape(cellfun(@pointInPolygon, holes), 1, []);

v = [offsetCage, offsetHoles];
vv = [v{1}; zeros( sum( cellfun(@numel, v(2:end))-2+1 ), 1 )];
unrefinedInwardCage = cage;

%% compute cauchy coordinates and its derivatives for samples
fSampleOnPolygon = @(n, p) subdivPolyMat(p, n)*p;

fPerimeter = @(x) sum( abs(x-x([2:end 1])) );
sumCagePerimeter = sum( cellfun(fPerimeter, v) );

fixedSamples = fSampleOnPolygon(numFixedSamples-1, cage);
% denseEvaluationSamples = fSampleOnPolygon(numDenseEvaluationSamples, cage);
activeSetPoolSamples = fSampleOnPolygon(numActiveSetPoolSamples, cage);

% [~, SODerivativeOfCauchyCoordinatesAtFixedSamples] = derivativesOfCauchyCoord(v, fixedSamples);

% energySamples = fSampleOnPolygon(nOutCageEngergySample, cage);
energySamples = cellfun(@(h) fSampleOnPolygon(ceil(fPerimeter(h)/sumCagePerimeter*numEnergySamples), h), [cage holes], 'UniformOutput', false);
nSamplePerCage = cellfun(@numel, energySamples);
energySamples = cat(1, energySamples{:});




denseEvaluationSamples = energySamples;



nextSampleInSameCage = [2:sum(nSamplePerCage) 1];   % next sample on the same cage, for Lipschitz constants and correct sample spacing computation
nextSampleInSameCage( cumsum(nSamplePerCage) ) = 1+cumsum( [0 nSamplePerCage(1:end-1)] );

%% compute D for interpolation
C = cauchyCoordinates(v, X, holeCenters);
D = derivativesOfCauchyCoord(v, X, holeCenters);


numFixedSamples = numel(fixedSamples);
numVirtualVertices = numel(vv);

needsPreprocessing = true;

%%
if ~exist('Phi', 'var') || numel(Phi)~=numVirtualVertices
    Phi = vv;
    Psy = Phi*0;
end

phipsyIters = [];
XP2PDeform = gather(C*Phi + conj(C*Psy));

%%
myinpoly = @(x, y) inpolygon(real(x), imag(x), real(y), imag(y));
assert( signedpolyarea( fC2R(offsetCage) ) > 0, 'cage vertex order reversed for proper Cauchy coordiates computation!');
if ~isempty(selfintersect(real(offsetCage), imag(offsetCage))), assert(false, 'source polygon has self-interesction'); end
% assert( all(myinpoly(w, offsetCage)), 'Boundary samples should be on offset inside the cage' );
% assert( isempty(selfintersect(real(w), imag(w))), 'offset polygon has self-interesction');

P2P_Deformation_Converged = 0;
