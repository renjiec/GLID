
P2Psrc = X(P2PVtxIds);
P2Pdst = P2PCurrentPositions;

datafile = fullfile(datadir, 'data.mat');
if exist(datafile, 'file') == 2
    warning( [datafile ' exists! backed up'] );
    movefile(datafile, [datafile '.bak']);
end

fprintf('saving data to %s\n', datafile);

if ~exist('P2Psets', 'var'), P2Psets = {struct('src', P2Psrc, 'dst', P2Pdst)}; end

if ~exist('textureClampingFlag', 'var'), textureClampingFlag = 0; end
if ~exist('textureScale', 'var'), textureScale = 1; end

save( datafile, 'allcages', 'Phi', 'Psy', 'P2Psrc', 'P2Pdst', 'cage_offset', 'numVirtualVertices', ...
    'numEnergySamples', 'textureScale', 'textureClampingFlag', 'P2Psets', 'img_w', 'img_h');

