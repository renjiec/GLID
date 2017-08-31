function M = subdivPolyMat(x, n)

% try to subdivide the polygon evenly, while keeping original vertices

assert(~isreal(x));

nx = numel(x);
% n = nx*2;

if n<=nx
    M = speye(nx, nx);
    return
end

edgelens = abs(x - x([2:end 1]));
edgesubs = ceil(edgelens/sum(edgelens)*n);
% edgesubs(1) = edgesubs(1)+1;

startingrows = cumsum([1;edgesubs]);
M = sparse(startingrows(end)-1, nx);
for i=1:nx
%     wt = [0; sort( rand(edgesubs(i)-1,1) )];
%     M(startingrows(i):startingrows(i+1)-1, [i mod(i,nx)+1]) = [1-wt wt];
    uniformCoef = (edgesubs(i):-1:1)'/edgesubs(i);
%     ss = 0/edgesubs(i);
%     uniformCoef(2:end) = sort( uniformCoef(2:end) + ( rand(edgesubs(i)-1,1)-0.5 )*ss, 'Descend' );
    M(startingrows(i):startingrows(i+1)-1, [i mod(i,nx)+1]) = [uniformCoef 1-uniformCoef];
end


% fSubdivMat = @(n, nsub) kron(eye(n), (nsub:-1:1)'/nsub) + kron(circshift(eye(n), -1), (0:nsub-1)'/nsub);
% M = fSubdivMat(nx, ceil(n/nx));