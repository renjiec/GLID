function d = distancePointToSegment(p, v1, v2)

complexDot = @(z1, z2) real(z1.*conj(z2));

t = complexDot(v2-v1, p-v1) ./ complexDot(v2-v1, v2-v1);

d = abs(p-((1-t).*v1 + t.*v2));

d_t0 = abs(p-v1);
d_t1 = abs(p-v2);

d(t<=0) = d_t0(t<=0);
d(t>=1) = d_t1(t>=1);