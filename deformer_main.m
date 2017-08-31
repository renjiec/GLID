fC2R = @(x) [real(x) imag(x)];
fR2C = @(x) complex(x(:,1), x(:,2));

%%
if ~exist('working_dataset', 'var') || isempty(working_dataset)
    working_dataset = 'giraffe';
    fprintf('working shape default: %s\n', working_dataset);
end

datasetbasedir = [cd '\data\'];
datadir = [datasetbasedir working_dataset '\'];

if exist('numMeshVertex', 'var')~=1 
    numMeshVertex = 10000;
    fprintf('numMeshVertex default: %d\n', numMeshVertex);
end

imgfilepath  = [datadir '\image.png'];
cagefilepath = [datadir '\cage.obj'];
datafile = [datadir '\data.mat'];

%%
if exist(cagefilepath, 'file') == 2
    % cage = fR2C( readObj( cagefilepath ) );
    [cx, cf] = readObj( cagefilepath, true );
    allcages = cellfun( @(f) fR2C(cx(f,:)), cf, 'UniformOutput', false );
else
    offset = 6;
    simplify = 8;
    allcages = GetCompCage(imgfilepath, offset, simplify, 0, 0);
end

P2Psrc = zeros(0,1); P2Pdst = zeros(0,1);

if exist(datafile, 'file') == 2
    load(datafile);
    hasDataLoaded = true;
end

for i=1:numel( allcages )
    curcage = allcages{i};
    % remove tail if repeating head
    if abs(curcage(1)-curcage(end))/sum( abs(curcage-curcage([2:end 1])) )<1e-4, curcage = curcage(1:end-1); end

    % make sure cage and holes have correct orientation
    if xor(i==1, signedpolyarea(curcage)>0); curcage = curcage(end:-1:1); end

    % make sure first two entries in the holes are sufficiently different, hole(2) - hole(1) \ne 0, 
    if i>1
        [~, imax] = max( abs(curcage-curcage([2:end 1])) );
        curcage = curcage([imax:end 1:imax-1]);
    end
    
    allcages{i} = curcage;
end

allcages = allcages(cellfun(@(x) numel(x)>2, allcages));

cage = allcages{1};
holes = reshape( allcages(2:end), 1, [] );

if ~exist('holes', 'var'), holes = {}; end

if exist([datadir 's.obj'], 'file') == 2
    fprintf('loading mesh from s.obj\n');
    [X, T] = readObj([datadir 's.obj']);
else
    [X, T] = cdt([cage, holes], [], numMeshVertex, false);
end

X = fR2C(X);

%% load handles
P2PVtxIds = triangulation(T, fC2R(X)).nearestNeighbor( fC2R(P2Psrc) );
P2PCurrentPositions = P2Pdst;

%% texture
img = imread(imgfilepath);
[w, h, ~] = size(img);
uv = fR2C([real(X)/h imag(X)/w])*100 + complex(0.5, 0.5);


%% deformation based on symmetrized Dirichlet energy minimization
harmonic_map_solvers = {'AQP', 'SLIM', 'Newton_SPDH', 'BDHM', 'cuNewton_SPDH'};

if hasGPUComputing                    
    default_harmonic_map_solver = harmonic_map_solvers{5};
else
    warning('no cuda capable GPU present, switching to CPU solver');
    default_harmonic_map_solver = harmonic_map_solvers{3};
end

harmonic_map_energies = {'ARAP', 'ISO', 'EISO'};
default_harmonic_map_energy = 'ISO';

iP2P = 1;
update_distortions_plots = false;
