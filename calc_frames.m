function [frames] = calc_frames(DerivativeOfCauchyCoordinates, Phi)

    fz = DerivativeOfCauchyCoordinates*Phi;
    assert(all(fz ~= 0));
    frames = abs(fz)./fz;

end
