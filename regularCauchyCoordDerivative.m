function D = regularCauchyCoordDerivative(x, z)

assert( ~isreal(z) && ~isreal(x), 'input should be complex!' );

n = numel(x);
nz = numel(z);

A = (eye(n)-circshift(eye(n), 1)) *x;

%% 
Bj = repmat(x.', nz, 1) - repmat(z(:), 1, n);
Bjp = Bj(:, [2:end 1]);
Bjm = Bj(:, [end 1:end-1]);
Aj = repmat(A.', nz, 1);
Ajp = Aj(:, [2:end 1]);
D = 1./Aj.*log(Bj./Bjm) - 1./Ajp.*log(Bjp./Bj);

D = D/complex(0, 2*pi);

if signedpolyarea( [real(x) imag(x)] ) < 0
    D = -D;
end
