%% set parameters

if exist('cage_offset', 'var')~=1
    fprintf('setting default parameters for p2p-harmonic deformation\n');
    cage_offset = 1e-1;
    numVirtualVertices = 1;
    numEnergySamples = 10000;
    p2p_weight = 1e5;
end

numDenseEvaluationSamples = numEnergySamples;

if exist('p2p_weight', 'var')~=1,       p2p_weight = 1e5;       end
if exist('numIterations', 'var')~=1,    numIterations = 3;      end
if exist('energy_parameter', 'var')~=1, energy_parameter = 1;   end
if exist('hessianSampleRate', 'var')~=1,hessianSampleRate = .1; end


%%
offsetCage = subdivPolyMat(cage, numVirtualVertices-sum( cellfun(@numel, holes) )-1)*polygonOffset(cage, -cage_offset, false);
offsetHoles = cellfun(@(h) polygonOffset(h, -cage_offset, false), holes, 'UniformOutput', false);
holeCenters = reshape(cellfun(@pointInPolygon, holes), 1, []);

v = [offsetCage, offsetHoles];
vv = [v{1}; zeros( sum( cellfun(@numel, v(2:end))-2+1 ), 1 )];
numVirtualVertices = numel(vv);

%% compute cauchy coordinates and its derivatives for samples
fSampleOnPolygon = @(n, p) subdivPolyMat(p, n)*p;

fPerimeter = @(x) sum( abs(x-x([2:end 1])) );
sumCagePerimeter = sum( cellfun(fPerimeter, v) );

energySamples = cellfun(@(h) fSampleOnPolygon(ceil(fPerimeter(h)/sumCagePerimeter*numEnergySamples), h), [cage holes], 'UniformOutput', false);
nSamplePerCage = cellfun(@numel, energySamples);
energySamples = cat(1, energySamples{:});

nextSampleInSameCage = [2:sum(nSamplePerCage) 1];   % next sample on the same cage, for Lipschitz constants and correct sample spacing computation
nextSampleInSameCage( cumsum(nSamplePerCage) ) = 1+cumsum( [0 nSamplePerCage(1:end-1)] );

%% compute D for interpolation
C = cauchyCoordinates(v, X, holeCenters);
D = derivativesOfCauchyCoord(v, X, holeCenters);

%% check if the offsetted boundary is simple
assert( signedpolyarea( fC2R(cage) ) > 0, 'outer boundary vertex should be ordered in CCW for proper Cauchy coordiates computation!');
assert( all(cellfun( @(h) signedpolyarea(fC2R(h)), holes )<0), 'inner boundary vertex should be ordered in CW for proper Cauchy coordiates computation!');
assert( ~any(cellfun(@(x) ~isempty( selfintersect(real(x), imag(x)) ), [offsetCage, offsetHoles])), 'offsetted outer/inner boundary should not selfintersects.');

%%
P2P_Deformation_Converged = 0;
NLO_preprocessed = false;

if ~exist('Phi', 'var') || numel(Phi)~=numVirtualVertices
    Phi = vv;
    Psy = Phi*0;
end

phipsyIters = [];
XP2PDeform = gather(C*Phi + conj(C*Psy));
needsPreprocessing = true;
