function A = signedAreas(x, t, planar)

if ~isreal(x), x = [real(x) imag(x)]; end
if nargin<3, planar = false; end

if size(x,2)==2 || range(x(:,3))<eps
    x(:,3) = 0;
    planar = true;
end

a = cross( x(t(:,1), :) - x(t(:,2), :), x(t(:,1), :) - x(t(:,3), :), 2 )/2;

if planar
    A = a(:,3);
else
    A = sqrt( sum(a.^2,2) );
end
