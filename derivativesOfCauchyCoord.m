function [D, E] = derivativesOfCauchyCoord(cage, z, holeCenters)
% Compute first and second derivatives of regular Cauchy coordiantes
%
% Input parameters:
% cage - cage
% z - points inside the cage

z = reshape(z, [], 1);

remove2coeffAtHoles = true;

if iscell(cage)    
    if nargout>1
        [D, E] = cellfun(@(c) derivativesOfCauchyCoord(c, z, holeCenters), cage, 'UniformOutput', false);
        E = cat(2, E{:});
    else
        D = cellfun(@(c) derivativesOfCauchyCoord(c, z, holeCenters), cage, 'UniformOutput', false);
    end

    D = cat(2, D{:});
    
    if numel(cage)>1
        % remove 2 holomorphic DOF for each hole
        if remove2coeffAtHoles
            cageSizes = cellfun(@numel, cage);
            flags = true(sum(cageSizes), 1);
            flags( reshape(cumsum(cageSizes(1:end-1)),[],1) + [1 2] ) = false;
            D = D(:, flags);
            if nargout>1, E = E(:, flags); end
        end

        assert(nargin>2 && ~isempty(holeCenters));
        
        holeCenters = reshape(holeCenters, 1, []);
        D = [D 1./(z-holeCenters)];
        
        if nargout>1, E = [E -(z-holeCenters).^-2]; end
    end
    
else


Aj = cage - cage([end 1:end-1], :);
Ajp = Aj([2:end 1], :);

Bj = bsxfun(@minus, cage.', z);
Bjp = Bj(:, [2:end 1]);
Bjm = Bj(:, [end 1:end-1]);


oneOver2pi_i = 1/(2*pi*1i);

D = oneOver2pi_i*(bsxfun(@rdivide, log(Bj./Bjm), Aj.') - bsxfun(@rdivide, log(Bjp./Bj), Ajp.'));

if(nargout > 1)
    E = oneOver2pi_i*(Bjp-Bjm)./(Bjm.*Bj.*Bjp);
end

end
