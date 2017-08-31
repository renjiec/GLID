function r = hasGPUComputing()

% r = false; return;

persistent hasgpu;
if isempty(hasgpu), hasgpu = gpuDeviceCount>0; end

r = hasgpu;