function [e, g, h] = harmonicMapAMIPSEnergyReal(D, phi, psy, mu, SPDHessian)

if nargin<5, SPDHessian = true; end

fz2 = abs(D*phi).^2;
gz2 = abs(D*psy).^2;

% mu = 2;

regularMIPS = false; % without exponent in the energy

if regularMIPS
    evec = (mu*2*(fz2+gz2)+1)./(fz2-gz2) + fz2-gz2;
else
    evec = exp( (mu*2*(fz2+gz2)+1)./(fz2-gz2) + fz2-gz2 );
end

e = sum( evec );

fz = D*phi;
gz = D*psy;

alpha1 =   1-(4*mu*gz2+1).*(fz2-gz2).^-2;
alpha2 = -(1-(4*mu*fz2+1).*(fz2-gz2).^-2);

if ~regularMIPS
    alpha1 = evec.*alpha1;
    alpha2 = evec.*alpha2;
end

if nargout>1
    fC2R = @(x) [real(x); imag(x)];
   
    g = 2*fC2R( [D'*(fz.*alpha1); D'*(gz.*alpha2)] );
end


if nargout>2
    m = numel(phi);
    DR = [real(D) -imag(D)];
    DI = [imag(D)  real(D)];
    DRDI = [DR; DI];
    n = size(D,1);
    
    if isa(D, 'gpuArray')
        h = gpuArray.zeros(m*4);
    else
        h = zeros(m*4);
    end

    %% 
    beta1 = 2*(4*mu*gz2+1).*(fz2-gz2).^-3;
    beta2 = 2*(4*mu*fz2+1).*(fz2-gz2).^-3;
    beta3 = -(beta1+beta2)/2;
    
    if ~regularMIPS
        beta1 = alpha1.^2./evec + beta1.*evec;
        beta2 = alpha2.^2./evec + beta2.*evec;
        beta3 = alpha1.*alpha2./evec + beta3.*evec;
    end

    if SPDHessian
        assert(false);  % need the general eigen factorization and modification from the paper
        i = alpha1<0;
        beta1(i) = beta1(i) + alpha1(i)/2./fz2(i);
        alpha1(i) = 0;
    end
    
    ss1 = [real(fz).^2; real(fz).*imag(fz); real(fz).*imag(fz); imag(fz).^2].*repmat(beta1,4,1) + [alpha1; zeros(n*2,1); alpha1]*.5;
    ss2 = [real(gz).^2; real(gz).*imag(gz); real(gz).*imag(gz); imag(gz).^2].*repmat(beta2,4,1) + [alpha2; zeros(n*2,1); alpha2]*.5;
    ss3 = [real(fz).*real(gz); real(fz).*imag(gz); imag(fz).*real(gz); imag(fz).*imag(gz)].*repmat(beta3,4,1);
    
    h(1:2*m, 1:2*m)             = DRDI'*[ DR.*ss1(1:n)+DI.*ss1(n+1:2*n); DR.*ss1(2*n+1:3*n)+DI.*ss1(3*n+1:4*n) ];
    h(end/2+1:end, end/2+1:end) = DRDI'*[ DR.*ss2(1:n)+DI.*ss2(n+1:2*n); DR.*ss2(2*n+1:3*n)+DI.*ss2(3*n+1:4*n) ];

    h(1:2*m, end/2+1:end)       = DRDI'*[ DR.*ss3(1:n)+DI.*ss3(n+1:2*n); DR.*ss3(2*n+1:3*n)+DI.*ss3(3*n+1:4*n) ];
    h(end/2+1:end, 1:2*m)       = h(1:2*m, end/2+1:end)';

    RIRI2RRII = [1:m m*2+(1:m) m+(1:m) m*3+(1:m)];
    h  = h(RIRI2RRII, RIRI2RRII)*4;
    

%     %%
%     h2 = h*0;
%     g2 = g*0;
%     for i=1:m*4
%         delta = 1e-6*rand;
%         if i<=m
%             [e1, g1] = harmonicMapAMIPSEnergyReal(D, phi+sparse(i,1,delta,m,1), psy, mu, SPDHessian);
%         elseif i<=m*2
%             [e1, g1] = harmonicMapAMIPSEnergyReal(D, phi, psy+sparse(i-m,1,delta,m,1), mu, SPDHessian);
%         elseif i<=m*3
%             [e1, g1] = harmonicMapAMIPSEnergyReal(D, phi+sparse(i-2*m,1,delta*1i,m,1), psy, mu, SPDHessian);
%         else
%             [e1, g1] = harmonicMapAMIPSEnergyReal(D, phi, psy+sparse(i-3*m,1,delta*1i,m,1), mu, SPDHessian);
%         end
% 
%         h2(:, i) = (g1-g)/delta;
%         g2(i) = (e1-e)/delta;
%     end
%     
%     
%     [norm(h2-h,'fro') norm(g2-g)]
end
