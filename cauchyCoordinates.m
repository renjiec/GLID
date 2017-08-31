function C = cauchyCoordinates(cage, z, holeCenters)
% Compute regular Cauchy coordiantes for given cage at strictly inside points z (z are not allowed to be on the boundary)
%
% Input parameters:
% cage - cage
% z - points inside the cage

z = reshape(z, [], 1);
remove2coeffAtHoles = true;

if iscell(cage)
    C = cellfun(@(c) cauchyCoordinates(c,z), cage, 'UniformOutput', false);    
    C = cat(2, C{:});
    
    % remove 2 holomorphic DOF for each hole
    if remove2coeffAtHoles
        cageSizes = cellfun(@numel, cage);
        flags = true(sum(cageSizes), 1);
        flags( reshape(cumsum(cageSizes(1:end-1)),[],1) + [1 2] ) = false;
        C = C(:, flags);
    end
    
    if numel(cage)>1
        assert(nargin>2 && ~isempty(holeCenters));
        
        holeCenters = reshape(holeCenters, 1, []);
        C = [C log( abs(z-holeCenters) )];  % a*2*log|z-z0| = a*log|z-z0| + conj( conj(a*log|z-z0|) ) = a*log(z-z0) + conj( conj(a*log(z-z0)) )
    end
else

Aj = cage - cage([end 1:end-1]);
Ajp = Aj([2:end 1], :);

Bj = bsxfun(@minus, cage.', z);
Bjp = Bj(:, [2:end 1]);
Bjm = Bj(:, [end 1:end-1]);


oneOver2pi_i = 1/(2*pi*1i);

%C = oneOver2pi_i*((Bjp./Ajp).*log(Bjp./Bj) - (Bjm./Aj).*log(Bj./Bjm));

C = oneOver2pi_i*(Bjp.*bsxfun(@rdivide, log(Bjp./Bj), Ajp.') - Bjm.*bsxfun(@rdivide, log(Bj./Bjm), Aj.'));

end