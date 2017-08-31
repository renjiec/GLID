function [L, indexOfMaxL] = computeLipschitzConstantOfDerivativeOfCauchy(v, samples)
%   v - vertices of the original cage.
%   samples - position of the vertices of the polygon. Each two consecutive vertices are considered as one segment (edge) of the polygon.
%             the first segment endpoints are samples(1) and samples(2).


    tic
 
    if iscell(v)
        cageSizes = cellfun(@numel, v);
        
        n = sum(cageSizes);
        inextv = [2:n 1];   % next phipsy in the same cage
        iprevv = [n 1:n-1]; % previous phipsy in the same cage
        inextv( cumsum(cageSizes) ) = 1+cumsum([0 cageSizes(1:end-1)]);
        iprevv( 1+cumsum([0 cageSizes(1:end-1)]) ) = cumsum(cageSizes);
        v = cat(1, v{:});
        z_j_minus_1 = v(iprevv);
        z_j_plus_1 = v(inextv);
    else  % simply connected domain
        z_j_minus_1 = circshift(v, 1);
        z_j_plus_1 = circshift(v, -1);
    end

    z_j = v;

    n = size(v, 1);
    m = size(samples, 1);

    if hasGPUComputing
        L = gpuArray.zeros(m, n);
    else
        L = zeros(m, n);
    end

    for i=1:n

        L(:,i) = computeLipschitzConstantOfDerivativeOfSingleCauchyBasisFunction(z_j_minus_1(i), z_j(i), z_j_plus_1(i), samples);

    end

% 
%     %the following code computes the index of the first two largest elements in each row of the matrix
%     [maximalValues, indexOfMaxL] = max(L, [], 2);
%     L(sub2ind([m,n], (1:m)', indexOfMaxL)) = -Inf; %temporarily replace the maximal value with -inf so we can search for the next largest elements 
%     [~, indexOfMaxL(:, 2)] = max(L, [], 2);
%     L(sub2ind([m,n], (1:m)', indexOfMaxL(:, 1))) = maximalValues;


    %the following code computes the index of the largest element in each row of the matrix and one of its neighbors (the largest among two)
    [~, indexOfMaxL] = max(L, [], 2);
    nextIndex = indexOfMaxL + 1;
    nextIndex(nextIndex > n) = 1;
    prevIndex = indexOfMaxL - 1;
    prevIndex(prevIndex < 1) = n;
    
    prevValues = L(sub2ind([m,n], (1:m)', prevIndex));
    nextValues = L(sub2ind([m,n], (1:m)', nextIndex));
    indicesWherePrevIndexIsLargest = find(prevValues > nextValues);
    indicesWhereNextIndexIsLargest = find(prevValues <= nextValues);
    
    indexOfMaxL(indicesWherePrevIndexIsLargest, 2) = prevIndex(indicesWherePrevIndexIsLargest);
    indexOfMaxL(indicesWhereNextIndexIsLargest, 2) = nextIndex(indicesWhereNextIndexIsLargest);

%	[maximalValues, indexOfMaxL] = max(L, [], 2);
%	indexOfMaxL(:, 2) = indexOfMaxL + 1;
%	indexOfMaxL(indexOfMaxL > n) = 1;

    fprintf('computeLipschitzConstantOfDerivativeOfCauchy preprocessing time: %.6f\n', toc);

end


function [L_j] = computeLipschitzConstantOfDerivativeOfSingleCauchyBasisFunction(z_j_minus_1, z_j, z_j_plus_1, v)


    v1 = v;
    v2 = circshift(v1, -1);
       
    d_j_minus_1 = distancePointToSegment(z_j_minus_1, v1, v2);
    d_j = distancePointToSegment(z_j, v1, v2);
    d_j_plus_1 = distancePointToSegment(z_j_plus_1, v1, v2);

    L_j = abs(z_j_plus_1 - z_j_minus_1)./(2*pi * d_j_minus_1.*d_j.*d_j_plus_1);

end


function [d] = distancePointToSegment(p, v1, v2)

    complexDot = @(z1, z2) real(z1.*conj(z2));
%     complexDot = @(z1, z2) real(z1).*real(z2)+imag(z1).*imag(z2);

    t = complexDot(v2-v1, p-v1) ./ complexDot(v2-v1, v2-v1);

	d = abs(p-((1-t).*v1 + t.*v2));
    
    d_t0 = abs(p-v1);
    d_t1 = abs(p-v2);
    
    d(t<=0) = d_t0(t<=0);
    d(t>=1) = d_t1(t>=1);

end

