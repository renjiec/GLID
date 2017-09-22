function t = maxtForPositiveArea(e0, e1)

if ~isreal(e0), e0 = [real(e0) imag(e0)]; end
if ~isreal(e1), e1 = [real(e1) imag(e1)]; end

A0 = e0(:, 1).*e0(:,4) - e0(:, 2).*e0(:,3);
A1 = e1(:, 1).*e1(:,4) - e1(:, 2).*e1(:,3);
B = dot(e0, [1 -1 -1 1].*e1(:,[4 3 2 1]), 2);

a = A0 + A1 - B;
b = B - 2*A0;
c = A0;



delta = b.^2 - 4*a.*c;
t = A0*inf;

% i = ~(a>0 & (delta<0 | b>0));
i = a<0 | (delta>0 & b<=0);
t(i) = (-b(i)-sqrt(delta(i)))./a(i)/2;


%% need to take care of special case where a==0 numerically
i2 = abs(a)<1e-16 & b<0;
t(i2) = -c(i2)./b(i2);

assert( ~any( a<0 & delta<0 ) );

t = max(t, 0);
