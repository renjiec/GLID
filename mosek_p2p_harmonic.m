function [status, energy, E_ARAP, E_POSITIONAL, phi, psy] = mosek_p2p_harmonic(...
    CauchyCoordinatesAtP2Phandles, DerivativeOfCauchyCoordinatesAtFixedSamples, ...
    DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1, DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2, DerivativeOfCauchyCoordinatesAtActiveSamples_k, ...
    P2PCurrentPositions, ARAP_g, ARAP_q, frames_fixed, frames_sigma2, frames_k, ...
    p2p_weight, sigma2_lower_bound, sigma1_upper_bound, k_upper_bound, ...
    numVirtualVertices, numFixedSamples, numEnergySamples, ...
    no_output)

%%
CCP2P = CauchyCoordinatesAtP2Phandles;
PosP2P = [real(P2PCurrentPositions); imag(P2PCurrentPositions)];
% DEnSample = DerivativeOfCauchyCoordinatesAtEnergySamples;
DFixed = DerivativeOfCauchyCoordinatesAtFixedSamples;
DSigma1 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1;
DSigma2 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2;
Dk = DerivativeOfCauchyCoordinatesAtActiveSamples_k;

mydiag = @(x) spdiags(reshape(x, [], 1), 0, size(x,1), size(x,1));
fC2RMatRe = @(x) [real(x) -imag(x)];
fC2RMatIm = @(x) [imag(x)  real(x)];
fC2RMat = @(x) [fC2RMatRe(x); fC2RMatIm(x)];
fC2RMatIthRowRe = @(x, i) [real(x(i,:)) -imag(x(i,:))];
fC2RMatIthRowIm = @(x, i) [imag(x(i,:))  real(x(i,:))];
fC2RMatIthRow = @(x, i) [fC2RMatIthRowRe(x, i); fC2RMatIthRowIm(x, i)];

nFix = size(DFixed, 1);
nSigma1 = size(DSigma1, 1);
nSigma2 = size(DSigma2, 1);
nK = size(Dk, 1);
% nEnSample = size(DEnSample, 1);
np = size(DFixed, 2);
nAuxS = nSigma1 + nFix;
sigma1 = sigma1_upper_bound;
sigma2 = sigma2_lower_bound;


fFrameMulDRe = @(frame, D) mydiag(real(frame))*fC2RMatRe(D) - mydiag(imag(frame))*fC2RMatIm(D);   % *phi = Re(frame*D*phi)
fFrameMulDIm = @(frame, D) mydiag(real(frame))*fC2RMatIm(D) + mydiag(imag(frame))*fC2RMatRe(D);   % *phi = Im(frame*D*phi)
fFrameMulD = @(frame, D) [ fFrameMulDRe(frame, D); fFrameMulDIm(frame, D) ];
                       
fFrameMulDIthRowRe = @(frame, D, i) real(frame(i))*fC2RMatRe(D(i,:)) - imag(frame(i))*fC2RMatIm(D(i,:));                       

nVar = 4*np + nAuxS;
                    
coneAcat = sparse(0, (nVar+1)*3);
if nFix>0
    coneAcat = [coneAcat;
            sparse(1:nFix, 4*np+(1:nFix), -1, nFix, nVar) ones(nFix,1)*sigma1 ...
            fC2RMatRe(DFixed) sparse(nFix, nVar+1-2*np) ...
            fC2RMatIm(DFixed) sparse(nFix, nVar+1-2*np);
            ...
            sparse(1:nFix, 4*np+(1:nFix),  1, nFix, nVar+1) ...
            sparse(nFix, 2*np) fC2RMatRe(DFixed) sparse(nFix, nAuxS+1) ...
            sparse(nFix, 2*np) fC2RMatIm(DFixed) sparse(nFix, nAuxS+1)];
end
if nSigma1>0
    coneAcat = [coneAcat;
            sparse(1:nSigma1, 4*np+nFix+(1:nSigma1),  1, nSigma1, nVar+1) ...
            fC2RMatRe(DSigma1) sparse(nSigma1, nVar+1-2*np) ...
            fC2RMatIm(DSigma1) sparse(nSigma1, nVar+1-2*np);
            ...
            sparse(1:nSigma1, 4*np+nFix+(1:nSigma1),  -1, nSigma1, nVar) ones(nSigma1,1)*sigma1...
            sparse(nSigma1, 2*np) fC2RMatRe(DSigma1) sparse(nSigma1, nAuxS+1) ...
            sparse(nSigma1, 2*np) fC2RMatIm(DSigma1) sparse(nSigma1, nAuxS+1)];
end
if nSigma2>0
    coneAcat = [coneAcat;
            fFrameMulDRe(frames_sigma2, DSigma2) sparse(nSigma2, nVar-2*np) ones(nSigma2, 1)*-sigma2 ...
            sparse(nSigma2, 2*np) fC2RMatRe(DSigma2) sparse(nSigma2, nAuxS+1) ...
            sparse(nSigma2, 2*np) fC2RMatIm(DSigma2) sparse(nSigma2, nAuxS+1)];
end
if nK>0
    coneAcat = [coneAcat;
            fFrameMulDRe(frames_k, Dk)*k_upper_bound sparse(nK, nVar-2*np+1) ...
            sparse(nK, 2*np) fC2RMatRe(Dk) sparse(nK, nAuxS+1) ...
            sparse(nK, 2*np) fC2RMatIm(Dk) sparse(nK, nAuxS+1)  ];
end

% coneAcat = [sparse(1:nFix, 4*np+(1:nFix), -1, nFix, nVar) ones(nFix,1)*sigma1 ...
%             fC2RMatRe(DFixed) sparse(nFix, nVar+1-2*np) ...
%             fC2RMatIm(DFixed) sparse(nFix, nVar+1-2*np);
%             ...
%             sparse(1:nFix, 4*np+(1:nFix),  1, nFix, nVar+1) ...
%             sparse(nFix, 2*np) fC2RMatRe(DFixed) sparse(nFix, nAuxS+1) ...
%             sparse(nFix, 2*np) fC2RMatIm(DFixed) sparse(nFix, nAuxS+1);
%             ...
%             sparse(1:nSigma1, 4*np+nFix+(1:nSigma1),  1, nSigma1, nVar+1) ...
%             fC2RMatRe(DSigma1) sparse(nSigma1, nVar+1-2*np) ...
%             fC2RMatIm(DSigma1) sparse(nSigma1, nVar+1-2*np);
%             ...
%             sparse(1:nSigma1, 4*np+nFix+(1:nSigma1),  -1, nSigma1, nVar) ones(nSigma1,1)*sigma1...
%             sparse(nSigma1, 2*np) fC2RMatRe(DSigma1) sparse(nSigma1, nAuxS+1) ...
%             sparse(nSigma1, 2*np) fC2RMatIm(DSigma1) sparse(nSigma1, nAuxS+1);
%             ...
%             fFrameMulDRe(frames_sigma2, DSigma2) sparse(nSigma2, nVar-2*np) ones(nSigma2, 1)*-sigma2 ...
%             sparse(nSigma2, 2*np) fC2RMatRe(DSigma2) sparse(nSigma2, nAuxS+1) ...
%             sparse(nSigma2, 2*np) fC2RMatIm(DSigma2) sparse(nSigma2, nAuxS+1);
%             ...
%             fFrameMulDRe(frames_k, Dk)*k_upper_bound sparse(nK, nVar-2*np+1) ...
%             sparse(nK, 2*np) fC2RMatRe(Dk) sparse(nK, nAuxS+1) ...
%             sparse(nK, 2*np) fC2RMatIm(Dk) sparse(nK, nAuxS+1)  ];

nCone = nFix*2+nSigma1*2+nSigma2+nK;
coneSizes = 3*ones(1, nCone);
coneAcat = reshape(coneAcat', nVar+1, [])';

% qobj = [ struct('quadform', true, 'wt', 1/numEnergySamples, 'A', [ fC2RMat( sqrtQ ) sparse(np*2, nVar-2*np+1); 
%                                                                    sparse(np*2, np*2) fC2RMat( sqrtQ ) sparse(np*2, nAuxS+1) ]), ...
%          struct('quadform', true, 'wt', p2p_weight, 'A', [ fC2RMat(CCP2P) [fC2RMatRe(CCP2P); -fC2RMatIm(CCP2P)] sparse(numel(PosP2P), nAuxS) -PosP2P]) ];

nq = size(ARAP_q,1);
qobj = [ struct('quadform', true, 'wt', 1/numEnergySamples, 'A', [ fC2RMat(ARAP_q) sparse(nq*2, nVar-2*np) -[real(ARAP_g) -imag(ARAP_g)]'; 
                                                                   sparse(nq*2, np*2) fC2RMat(ARAP_q) sparse(nq*2, nAuxS+1) ]), ...
         struct('quadform', true, 'wt', p2p_weight, 'A', [ fC2RMat(CCP2P) [fC2RMatRe(CCP2P); -fC2RMatIm(CCP2P)] sparse(numel(PosP2P), nAuxS) -PosP2P]) ];

param = [];
% param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-10;
% param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-10;
% param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-10;
% param.MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1.4901e-08;

option = struct('dualize', true, 'param', param, 'info', 3);
if no_output, option.info=0; end
    
A = [sparse(1:2, np*(2:3)+1, 1, 2, nVar); ...
     fFrameMulDRe(frames_fixed, DFixed)               sparse(nFix, np*2) -speye(nFix, nAuxS); ...
     fFrameMulDRe(frames_fixed, DFixed)*k_upper_bound sparse(nFix, np*2) -speye(nFix, nAuxS) ];
blc = [zeros(2,1); ones(nFix,1)*sigma2_lower_bound; zeros(nFix,1)];
buc = [zeros(2,1); ones(nFix*2,1)*inf];

% c = -2*[real(arap_frames_vector) -imag(arap_frames_vector) zeros(1,nAuxS+2*np)];
% [x, en, status] = buildMosekSOCP( qobj(1).wt*c, qobj, struct('RotatedCone', false(1, nCone), 'A', coneAcat, 'coneSizes', coneSizes), ...
%                           A, blc, buc, option );

[x, en, status] = buildMosekSOCP( zeros(1, nVar), qobj, struct('RotatedCone', false(1, nCone), 'A', coneAcat, 'coneSizes', coneSizes), ...
                          A, blc, buc, option );

energy = en(1) + 1 - norm(ARAP_g)^2/numEnergySamples;
% E_ARAP = (en(2)+c*x)*qobj(1).wt + 1;
% E_POSITIONAL = en(3)*qobj(2).wt;

%%
phi = complex(x(1:np), x(np+(1:np)));
psy = complex(x(2*np+(1:np)), x(np*3+(1:np)));

% fz_fixed = DerivativeOfCauchyCoordinatesAtFixedSamples*phi;
% fzbar_fixed = conj(DerivativeOfCauchyCoordinatesAtFixedSamples*psy);
% fz_sigma1 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1*phi;
% fzbar_sigma1 = conj(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma1*psy);
% fz_sigma2 = DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2*phi;
% fzbar_sigma2 = conj(DerivativeOfCauchyCoordinatesAtActiveSamplesSigma2*psy);
% fz_k = DerivativeOfCauchyCoordinatesAtActiveSamples_k*phi;
% fzbar_k = conj(DerivativeOfCauchyCoordinatesAtActiveSamples_k*psy);
% E_ARAP = 1/numEnergySamples*(quad_form(phi, Q) + quad_form(psy, Q) - 2*real(arap_frames_vector*phi) + numEnergySamples); %ARAP energy

fSumSquares = @(x) sum(x.^2);
E_ARAP = 1+1/numEnergySamples*( fSumSquares(ARAP_q*phi-ARAP_g') - norm(ARAP_g)^2 + fSumSquares(ARAP_q*psy) ); %ARAP energy
E_POSITIONAL = p2p_weight*fSumSquares( abs(CauchyCoordinatesAtP2Phandles*phi + conj(CauchyCoordinatesAtP2Phandles*psy) - P2PCurrentPositions) ); %P2P energy

%%
% max( [abs(fz_fixed)+abs(fzbar_fixed) - sigma1_upper_bound
% abs(fzbar_fixed) - real(fz_fixed.*frames_fixed) + sigma2_lower_bound
% abs(fzbar_fixed) - k_upper_bound*real(fz_fixed.*frames_fixed)
% abs(fz_sigma1) + abs(fzbar_sigma1) - sigma1_upper_bound
% abs(fzbar_sigma2) - real(fz_sigma2.*frames_sigma2) + sigma2_lower_bound;
% abs(fzbar_k) - k_upper_bound*real(fz_k.*frames_k) ])
% E_ARAP+E_POSITIONAL
% energy
% E_ARAP+E_POSITIONAL-energy
