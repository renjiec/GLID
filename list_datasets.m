
datadirs = dir('data');
isdataset = arrayfun(@(i) datadirs(i).isdir && datadirs(i).name(1)~='.', 1:numel(datadirs));
datasets = {datadirs(isdataset).name};
