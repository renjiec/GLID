function t = maxtForPhiPsy( fz, gz, dfz, dgz )

a = abs(dfz)^2 - abs(dgz)^2;
b = 2*real( conj(fz)*dfz - conj(gz)*dgz );
c = abs(fz)^2-abs(gz)^2;

% assert( all( c>=-1e-10 ) );

delta = b^2 - 4*a*c;


% t = single(inf);
t = inf;
if ~(a>0 && (delta<0 || b>0))
    t = (-b-sqrt(delta))/a/2;
end

% assert( t>=0 );
t = max(t, 0);
