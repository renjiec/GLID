function c = pointInPolygon(b)

fR2C = @(x) complex(x(:,1), x(:,2));
if isreal(b), b=fR2C(b); end

dt = delaunayTriangulation(real(b), imag(b), [1:numel(b); 2:numel(b) 1]');

cc = fR2C(dt.circumcenter);
d = abs( b(dt(:,1)) - cc );
d( ~inpolygon(real(cc), imag(cc), real(b), imag(b)) ) = 0;

[~, i] = max(d);
c = cc(i);