function [e, g, h] = harmonicMapIsometryicEnergyReal(D, phi, psy, epow, SPDHessian)

if nargin<4, epow = 1;          end
if nargin<5, SPDHessian = true; end


fz2 = abs(D*phi).^2;
gz2 = abs(D*psy).^2;

% evec = abs( 2*(fz2 + gz2).*(1+(fz2 - gz2).^-2)-4 );  % abs is for avoid numerical issue when e is exactly 0
evec = 2*(fz2 + gz2).*(1+(fz2 - gz2).^-2);  % use original definition for the moment
e = sum( evec.^epow );

if nargout>1
    s1 = 1-((fz2-gz2).^-3).*(fz2+3*gz2);
    s2 = 1+((fz2-gz2).^-3).*(3*fz2+gz2);

%     s1 = max(s1, 0);      % should the grad be modified too?
    epp = epow*evec.^(epow-1);
    
    s1a = s1; s2a = s2;
    s1 = s1.*epp; s2 = s2.*epp;
    
    
    fC2R = @(x) [real(x); imag(x)];
    g = 4*fC2R( [D'*(D*phi.*s1); D'*(D*psy.*s2)] );  
end

if nargout>2
    
%     if SPDHessian
% %         fprintf('\n#s1<0: %d\n', sum( s1<0 ));
%         s1 = max(s1, 0);
% %         s1a= max(s1a, 0);
%     end
    
    m = numel(phi);
    
    s3 = 4*(fz2 + 5*gz2).*(fz2-gz2).^-4;
    s4 = 4*(5*fz2 + gz2).*(fz2-gz2).^-4;
    s5 = 12*(fz2+gz2).*(fz2-gz2).^-4;

    s3 = s3.*epp; s4 = s4.*epp; s5 = s5.*epp;

    fC2Rm = @(m) [real(m) -imag(m); imag(m) real(m)];
    fM2M = @(m) fC2Rm( complex(m(1:end/2, 1:end/2), -m(1:end/2, end/2+1:end)) ); 
    fIsComplexMat = @(m) norm(m-fM2M(m), 'fro');     % for checking if m is a complex matrix represented in real numbers
    
    fz = D*phi;
    gz = D*psy;

    %%	version a    
%     h = blkdiag( fC2Rm( D'*bsxfun(@times, D, s1) ), fC2Rm( D'*bsxfun(@times, D, s2) ) );
%     DR = bsxfun(@times, [real(D) -imag(D)], s3.^0.5.*real(fz));
%     DI = bsxfun(@times, [imag(D)  real(D)], s3.^0.5.*imag(fz));
%     h(1:2*m, 1:2*m) = h(1:2*m, 1:2*m) + [DR' DI' DR' DI']*[DR; DI; DI; DR];
%     
%     DR = bsxfun(@times, [real(D) -imag(D)], s4.^0.5.*real(gz));
%     DI = bsxfun(@times, [imag(D)  real(D)], s4.^0.5.*imag(gz));
%     h(end/2+1:end, end/2+1:end) = h(end/2+1:end, end/2+1:end) + [DR' DI' DR' DI']*[DR; DI; DI; DR];
%     
%     DRf = bsxfun(@times, [real(D) -imag(D)], s5.^0.5.*real(fz));
%     DRg = bsxfun(@times, [real(D) -imag(D)], s5.^0.5.*real(gz));
%     DIf = bsxfun(@times, [imag(D)  real(D)], s5.^0.5.*imag(fz));
%     DIg = bsxfun(@times, [imag(D)  real(D)], s5.^0.5.*imag(gz));
%     
%     h(1:2*m, end/2+1:end) =  -[DRf' DIf' DRf' DIf']*[DRg; DIg; DIg; DRg];
%     h(end/2+1:end, 1:2*m) =  h(1:2*m, end/2+1:end)';

%%
    DR = [real(D) -imag(D)];
    DI = [imag(D)  real(D)];
    DRDI = [DR; DI];
    n = numel(s1);
    
    if isa(D, 'gpuArray')
        h = gpuArray.zeros(m*4);
    else
        h = zeros(m*4);
    end

    %% version b
%     ss1 = [real(fz).^2; imag(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz)].*repmat(s3,4,1) + [s1; s1; zeros(n*2,1)];
%     ss2 = [real(gz).^2; imag(gz).^2; real(gz).*imag(gz); real(gz).*imag(gz)].*repmat(s4,4,1) + [s2; s2; zeros(n*2,1)];
%     ss3 = [real(fz).*real(gz); imag(fz).*imag(gz); real(fz).*imag(gz); real(gz).*imag(fz)].*repmat(s5,4,1);
%     h2(1:2*m, 1:2*m)             = [DR' DI' DR' DI']*( [DR; DI; DI; DR].*ss1 );
%     h2(end/2+1:end, end/2+1:end) = [DR' DI' DR' DI']*( [DR; DI; DI; DR].*ss2 );
%     h2(1:2*m, end/2+1:end)       =-[DR' DI' DR' DI']*( [DR; DI; DI; DR].*ss3 );
%     h2(end/2+1:end, 1:2*m)       = h2(1:2*m, end/2+1:end)';

    %% version c, in sync with cuda version
%     ss1 = [real(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz); imag(fz).^2].*repmat(s3,4,1) + [s1; zeros(n*2,1); s1];
%     ss2 = [real(gz).^2; real(gz).*imag(gz); real(gz).*imag(gz); imag(gz).^2].*repmat(s4,4,1) + [s2; zeros(n*2,1); s2];
%     ss3 = [real(fz).*real(gz); real(fz).*imag(gz); imag(fz).*real(gz); imag(fz).*imag(gz)].*repmat(-s5,4,1);
%     h(1:2*m, 1:2*m)             = DRDI'*[ DR.*ss1(1:n)+DI.*ss1(n+1:2*n); DR.*ss1(2*n+1:3*n)+DI.*ss1(3*n+1:4*n) ];
%     h(end/2+1:end, end/2+1:end) = DRDI'*[ DR.*ss2(1:n)+DI.*ss2(n+1:2*n); DR.*ss2(2*n+1:3*n)+DI.*ss2(3*n+1:4*n) ];
%     h(1:2*m, end/2+1:end)       = DRDI'*[ DR.*ss3(1:n)+DI.*ss3(n+1:2*n); DR.*ss3(2*n+1:3*n)+DI.*ss3(3*n+1:4*n) ];
%     h(end/2+1:end, 1:2*m)       = h(1:2*m, end/2+1:end)';
% 
% %     h = harmonic_newton_r0(D, [phi psy]);
% %     norm( h2 - h, 'fro')
% 
%     RIRI2RRII = [1:m m*2+(1:m) m+(1:m) m*3+(1:m)];
%     h  = h(RIRI2RRII, RIRI2RRII)*4;
%     
%     
%     if epow~=1
%         gpp = fC2R( [D.*(conj(fz).*s1a) D.*(conj(gz).*s2a)]'*4 ).*sqrt(evec'.^(epow-2));
%     %     g == fC2R( sum( gpp,2 ) );
%         h = epow*h + gpp*gpp'*epow*(epow-1);
%     end
    
    
    %% version d, merge powered energy computation for better performance
    w = epow*(epow-1)*evec.^(epow-2)*4; % note that 4 appears on both sides of gpp(as out product) in the above version
    w = min( w, 1e10 );  % when evec is 0 and epow<2, w becomes infinite!
%     if epow == 1, w=0; end
    
%     w = 0; s3(:) = 1; s4(:) = 1; s5(:) = 1; s1(:) = 0; s2(:) = 0;
%     DRDI2 = blkdiag(DRDI, DRDI);
%     fg = [real(fz) imag(fz) -real(gz) -imag(gz)];
%     
%     DRDI3 = [[real(fz) imag(fz)]*DRDI -[real(gz) imag(gz)]*DRDI];
%     h2 = DRDI3' * DRDI3;
    
    
    if SPDHessian
%         fprintf('\n#s1<0: %d\n', sum( s1<0 ));
%         s1 = max(s1, 0);
% %         s1a= max(s1a, 0);
        i = s1<0;
        s3(i) = s3(i) + s1(i)./fz2(i);
        s1(i) = 0;
    end

    ss1 = [real(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz); imag(fz).^2].*repmat(s3 + w.*s1a.^2,4,1) + [s1; zeros(n*2,1); s1];
    ss2 = [real(gz).^2; real(gz).*imag(gz); real(gz).*imag(gz); imag(gz).^2].*repmat(s4 + w.*s2a.^2,4,1) + [s2; zeros(n*2,1); s2];
    ss3 = [real(fz).*real(gz); real(fz).*imag(gz); imag(fz).*real(gz); imag(fz).*imag(gz)].*repmat(-s5 + w.*s1a.*s2a,4,1);
%     h(1:2*m, 1:2*m)             = DRDI'*[ DR.*ss1(1:n)+DI.*ss1(n+1:2*n); DR.*ss1(2*n+1:3*n)+DI.*ss1(3*n+1:4*n) ];
%     h(end/2+1:end, end/2+1:end) = DRDI'*[ DR.*ss2(1:n)+DI.*ss2(n+1:2*n); DR.*ss2(2*n+1:3*n)+DI.*ss2(3*n+1:4*n) ];
%     h(1:2*m, end/2+1:end)       = DRDI'*[ DR.*ss3(1:n)+DI.*ss3(n+1:2*n); DR.*ss3(2*n+1:3*n)+DI.*ss3(3*n+1:4*n) ];

    fmydgmm = @(s) bsxfun(@times, DR, s(1:n)) + bsxfun(@times, DI, s(n+1:n*2));
    h(1:2*m, 1:2*m)             = DRDI'*[ fmydgmm(ss1(1:n*2)); fmydgmm(ss1(n*2+1:end)) ];
    h(end/2+1:end, end/2+1:end) = DRDI'*[ fmydgmm(ss2(1:n*2)); fmydgmm(ss2(n*2+1:end)) ];
    h(1:2*m, end/2+1:end)       = DRDI'*[ fmydgmm(ss3(1:n*2)); fmydgmm(ss3(n*2+1:end)) ];
    
    h(end/2+1:end, 1:2*m)       = h(1:2*m, end/2+1:end)';
    
    
%     ftomat = @(x) [x([1 3]) x([2 4])];
%     S = [ftomat(ss1) ftomat(ss3); ftomat(ss3)' ftomat(ss2)];
%     h2 = blkdiag(DRDI, DRDI)' * S * blkdiag(DRDI, DRDI);
%     ev = real( [eig(S) [s1; 1+3*(abs(fz)+abs(gz))^-4; s2; 1+3*(abs(fz)-abs(gz))^-4]] )
%     
%     ss1a = [real(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz); imag(fz).^2].*repmat(s3 + s1/fz2, 4, 1);
%     h2a = blkdiag(DRDI, DRDI)' * [ftomat(ss1a) ftomat(ss3); ftomat(ss3)' ftomat(ss2)] * blkdiag(DRDI, DRDI);
%     ev2 = sort( real( [eig(h2) eig(h2a)] ) )
    
%     h = h*4;
    RIRI2RRII = [1:m m*2+(1:m) m+(1:m) m*3+(1:m)];
    h  = h(RIRI2RRII, RIRI2RRII)*4;
end
