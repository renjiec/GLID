function p = polygonOffset(x, d, useRelatived)

if nargin<3
    useRelatived = true;
end

% dRelative = .2;
% x = mxsub*x0;

complexInput = ~isreal(x);
if complexInput
    x = [real(x) imag(x)];
end
n = size(x,1);
fRowNorm = @(x) sqrt( sum(x.^2, 2) );

%% take care of straight edges
e = x - x([2:end 1],:);
e = complex(e(:,1), e(:,2));
angles = angle(e);
iscorner = abs( angles - angles([end 1:end-1]) ) > 1e-8;

if any(~iscorner)
    cornids = find(iscorner);
    ncorn = numel(cornids);
    
    prevcorn = cumsum(iscorner);
    prevcorn(prevcorn==0) = ncorn;
    
    prevcornidsX = cornids(prevcorn);
    nextcornidsX = cornids(mod(prevcorn, ncorn)+1);

    fDistVij = @(i, j) fRowNorm( x(i,:) - x(j,:) );
    wts = fDistVij(1:n, nextcornidsX)./fDistVij(prevcornidsX, nextcornidsX);
    
    S = sparse( [1:n 1:n], [prevcorn mod(prevcorn, ncorn)+1], [wts; 1-wts]', n, ncorn);

    x = x(cornids,:);
    n = numel(cornids);
end

%% offset a simple polygon
offs = zeros(n, 2);

i2 = 1:n;
i1 = [n 1:n-1];
i3 = [2:n 1];
A = x(i1,1).*(x(i2,2)-x(i3,2)) + x(i2,1).*(x(i3,2)-x(i1,2)) + x(i3,1).*(x(i1,2)-x(i2,2));
for i=1:n
    e1 = x(i, :) - x(mod(i, n)+1, :);
    e2 = x(mod(i+n-2, n)+1, :) - x(i, :);
    offs(i,:) = [-e1' e2']*fRowNorm([e2; e1])/A(i);
end

if useRelatived
	d = min( fRowNorm(x-x([2:end 1],:)) )*d;
end

noff = numel(d);
p = repmat(x, 1, noff) + reshape([offs(:,1)*d; offs(:,2)*d], n, []);

if complexInput
    p = complex( p(:,1:2:end), p(:,2:2:end) );
end

%%
if any(~iscorner)
    p = S*p;
end

% fDrawPolygon = @(x) plot(x([1:end 1],1:2:end), x([1:end 1],2:2:end), '-*');
% figuredocked; fDrawPolygon(x); axis equal;
% hold on; h = fDrawPolygon(p);
% set(h, 'color', 'r'); axis equal;
% 
% shownumber(x, 1:n, 'b');
% shownumber(p);
