function [theta, thetaDiffs] = extractBestAngle(fz)
%given fz on the boundary of a polygon (at a set of samples) we compute the angle theta at each sample such that exp(1i*theta)=exp(1i*angle(fz)).
%note that adding 2*pi*k to each sample (where k is an integer) doesn't affect the correctness of the above equation.
%however, we would like to set k at each sample automatically such that the difference between each two consecutive angles is minimized.

    thetaDiffs = angle(fz./fz([end 1:end-1]));
    theta = angle(fz(end)) + cumsum(thetaDiffs);
    thetaDiffs(1) = theta(1) - theta(end);
%     %sanity check
%     err = norm(exp(1i*theta)-exp(1i*angle(fz)), Inf);
%     assert(err < 1e-10);
    
end
