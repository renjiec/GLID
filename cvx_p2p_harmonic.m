function [status, optval, E_ARAP, E_POSITIONAL, phi, psy] = cvx_p2p_harmonic(...
    CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
    DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
    P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
    p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
    numVirtualVertices, numFixedSamples, numEnergySamples, ...
    no_output, forceConformalMode)

if no_output
    cvx_begin quiet
else
    cvx_begin
end
    cvx_solver Mosek
    %cvx_solver Gurobi
    %cvx_solver_settings('QCPDual', 1);
    %cvx_solver SDPT3
	%cvx_solver SeDuMi

    variable phi(numVirtualVertices, 1) complex

    if forceConformalMode
        psy = zeros(numVirtualVertices, 1); %force psy to be zero and don't use psy as variable - this will produce a conformal mapping
    else
        variable psy(numVirtualVertices, 1) complex
    end

    variable t(numFixedSamples, 1)
    variable s(numFixedSamples, 1)

    fz_fixed = DerivativeOfCauchyCoordinatesAtFixedSamples*phi;
    fzbar_fixed = conj(DerivativeOfCauchyCoordinatesAtFixedSamples*psy);

    fz_sigma1 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1*phi;
    fzbar_sigma1 = conj(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1*psy);
    
    fz_sigma2 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2*phi;
    fzbar_sigma2 = conj(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2*psy);

	fz_k = DerivativeOfCauchyCoordinatesAtActiveSamples_k*phi;
	fzbar_k = conj(DerivativeOfCauchyCoordinatesAtActiveSamples_k*psy);

    E_ARAP = 1+1/numEnergySamples*( sum_square_abs(ARAP_q*phi-ARAP_g') - norm(ARAP_g)^2 + sum_square_abs(ARAP_q*psy) ); %ARAP energy    
%     E_ARAP = 1/numEnergySamples*(quad_form(phi, Q) + quad_form(psy, Q) - 2*real(arap_frames_vector*phi) + numEnergySamples); %ARAP energy
%     E_POSITIONAL = p2p_weight*sum(abs(CauchyCoordinatesAtP2Phandles*phi + conj(CauchyCoordinatesAtP2Phandles*psy) - P2PCurrentPositions)); %P2P energy
    E_POSITIONAL = p2p_weight*sum_square_abs(CauchyCoordinatesAtP2Phandles*phi + conj(CauchyCoordinatesAtP2Phandles*psy) - P2PCurrentPositions); %P2P energy

    minimize E_ARAP + E_POSITIONAL;
    
    subject to
        %CauchyCoordinatesAtP2Phandles*phi + conj(CauchyCoordinatesAtP2Phandles*psy) == P2PCurrentPositions; %hard constraints

        psy(1) == 0; %to make the decomposition f(z)=phi+conj(psy) unique
%         psy == 0; %temporarily force the result to be conformal

        if(0) %switch from 0 to 1 to use a reduced subspace
            %reduced subspace for holomorphic part
            (real(CauchyCoordinatesAtVertices) - eye(size(CauchyCoordinatesAtVertices)))*real(phi) == imag(CauchyCoordinatesAtVertices)*imag(phi);

            %reduced subspace for anti-holomorphic part
            (real(CauchyCoordinatesAtVertices) - eye(size(CauchyCoordinatesAtVertices)))*real(psy) == imag(CauchyCoordinatesAtVertices)*imag(psy);
        end
        
        %for fixed samples we use all three types of constraints: sigma1, sigma2, and k
        abs(fz_fixed) <= t; %standard 2nd order convex cone
        abs(fzbar_fixed) <= s; %standard 2nd order convex cone
        s <= sigma1_upper_bound - t; %linear inequality
        s <= real(fz_fixed.*frames_fixed) - sigma2_lower_bound; %linear inequality
        s <= k_upper_bound*real(fz_fixed.*frames_fixed); %linear inequality

        
        %these are constraints at active points for sigma2 only
        if(~isempty(fz_sigma1))
            abs(fz_sigma1) + abs(fzbar_sigma1) <= sigma1_upper_bound;
        end
        
        %these are constraints at active points for sigma2 only
        if(~isempty(fz_sigma2))
            abs(fzbar_sigma2) <= real(fz_sigma2.*frames_sigma2) - sigma2_lower_bound;
        end

        %these are constraints at active points for k only
        if(~isempty(fz_k))
            abs(fzbar_k) <= k_upper_bound*real(fz_k.*frames_k);
        end
        
cvx_end

status = cvx_status;
optval = cvx_optval;

end


    