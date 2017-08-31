function numZeros = computeNumOfZerosOf_fz_inside(cage, phi, insidePolygon)
tic;
argumentPrincipleIntegral = integralOfDerivativeOfLog_fz_onClosedSimplePolygon(cage, phi, insidePolygon); %this should return a real integer
numZeros = round(real(argumentPrincipleIntegral/(2*pi*1i)));
assert(numZeros >= 0); %it should never be negative
fprintf('computeNumOfZerosOf_fz_inside time: %.5f\n', toc);
end



function argumentPrincipleIntegral = integralOfDerivativeOfLog_fz_onClosedSimplePolygon(cage, phi, insidePolygon)
%insidePolygon must be fully contained in the cage (which is also a polygon).
%for example insidePolygon can be an inward offset of the cage polygon.

assert(size(insidePolygon, 1) >= 3);

argumentPrincipleIntegral = integral(@(z) derivativeOfLog_fz(cage, z, phi), insidePolygon(1, 1), insidePolygon(1, 1), 'Waypoints', insidePolygon(2:end));
%argumentPrincipleIntegral = integral(@(z) derivativeOfLog_fz(cage, z, phi), insidePolygon(1, 1), insidePolygon(1, 1), 'Waypoints', insidePolygon(2:end), 'AbsTol',0.1);
%argumentPrincipleIntegral = integral(@(z) derivativeOfLog_fz(cage, z, phi), insidePolygon(1, 1), insidePolygon(1, 1), 'Waypoints', insidePolygon(2:end), 'AbsTol',1e-6);

end


function r = derivativeOfLog_fz(cage, z, phi)
%z can be a scalar or vector of complex points

[D, E] = derivativesOfCauchyCoord(cage, z);
%size(z)
r = (E*phi)./(D*phi);
r = gather(r.');

end

