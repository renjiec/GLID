function [e, g, h] = harmonicMapExpIsometryicEnergy(D, phi, psy, s, SPDHessian)

if nargin<4, s = 1; end
if nargin<5, SPDHessian = true; end

fz2 = abs(D*phi).^2;
gz2 = abs(D*psy).^2;

    

evec = exp( s*(fz2 + gz2).*(1+(fz2 - gz2).^-2) );
e = sum( evec );

fz = D*phi;
gz = D*psy;

alpha1 = (1-((fz2-gz2).^-3).*(fz2+3*gz2)).*evec*s;
alpha2 = (1+((fz2-gz2).^-3).*(3*fz2+gz2)).*evec*s;

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
    beta1 = ((2*fz2 + 10*gz2).*(fz2-gz2).^-4).*evec*s + (alpha1).^2./evec;
    beta2 = ((10*fz2 + 2*gz2).*(fz2-gz2).^-4).*evec*s + (alpha2).^2./evec;
    beta3 =-(6*(fz2+gz2).*(fz2-gz2).^-4).*evec*s + alpha1.*alpha2./evec;
    
    if SPDHessian
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
end
