function A = signedpolyarea(x, y)

if nargin<2
    if ~isreal(x)
        x = [real(x) imag(x)];
    end
    y = x(:,2);
    x = x(:,1);
end

A = sum( x.*y([2:end 1]) - x([2:end 1]).*y )/2;
