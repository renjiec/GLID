function A = signedAreas(x, t, planar)

if ~isreal(x)
    x = [real(x) imag(x)];
end

if nargin<3
    planar = false;
end

if size(x,2)==2 || range(x(:,3))<eps
    x(:,3) = 0;
    planar = true;
end

% bug fixed 7/24/2011. 
% C = CROSS(A,B) returns the cross product of A and B along the first dimension of length 3
a = cross( x(t(:,1), :) - x(t(:,2), :), x(t(:,1), :) - x(t(:,3), :), 2 )/2;

if planar
    A = a(:,3);
%     if sum(A) < 0
%         A = -A;
%     end
else
    A = sqrt( sum(a.^2,2) );
end
