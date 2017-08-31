function [x, en, status, prob, dualprob] = buildMosekSOCP(c, qobj, cones, a, bl, bu, option)

if nargin<6
    bu = bl;
end

if nargin<7
    option = struct('dualize', false, 'param', [], 'info', 0);
end

qEnergyVarId = [];
nc0 = numel(c);

energyInSqrt = false;
%% convert sum of squares to norm
if ~any(c) && all( cat(1, qobj.quadform) )
    nq = numel(qobj);
    c(end+1) = 1;
    
    nVar = numel(c);
    
%     newA = cat(1, qobj.A);
    newA = qobj(1).A*sqrt(qobj(1).wt);
    for i=2:nq
        newA = [newA; qobj(i).A*sqrt(qobj(i).wt)];
    end
    
    
    cones.A = [ cones.A(:, 1:end-1) sparse(size(cones.A,1),1) cones.A(:,end);
                sparse(1, nVar, 1, 1, nVar+1);
                newA(:, 1:end-1) sparse(size(newA,1), 1) newA(:, end) ];

    cones.RotatedCone(end+1) = false;
    cones.coneSizes(end+1) = size(newA,1)+1;
    a(:, end+1) = 0;

    qobj = [];
    energyInSqrt = true;
end



%% quadratic energy, convert to cone+linear constraints
for i=1:numel(qobj)
    nVar = numel(c);

    A = qobj(i).A;
    [m, n] = size(A);
    if qobj(i).quadform
        nNewVar = 2;
        cones.A = [ cones.A(:, 1:end-1) sparse(size(cones.A,1),2) cones.A(:,end);
                    sparse(1:2, nVar+(1:2), 1, 2, nVar+nNewVar+1);
                    A(:,1:end-1) sparse(m, nNewVar+nVar+1-n) A(:,end) ];
        cones.RotatedCone(end+1) = true;
        cones.coneSizes(end+1) = m+2;
        
        c(end+(1:2)) = [qobj(i).wt 0];
        a = [   a sparse(size(a,1), nNewVar); ...
                sparse(1, nVar+2, 1, 1, nVar+nNewVar) ];
        bl(end+1) = 0.5;
        bu(end+1) = 0.5;
    else
        nNewVar = 1;
        cones.A = [ cones.A(:, 1:end-1) sparse(size(cones.A,1),1) cones.A(:,end);
                    sparse(1, nVar+1, 1, 1, nVar+nNewVar+1);
                    A(:,1:end-1) sparse(m, nNewVar+nVar+1-n) A(:,end) ];
        cones.RotatedCone(end+1) = false;
        cones.coneSizes(end+1) = m+1;
        
        c(end+1) = qobj(i).wt;
        a = [ a sparse(size(a,1), nNewVar) ];
    end
    
    qEnergyVarId(end+1) = nVar + 1;
end

nVar0 = numel(c);

if ~isempty(qobj)
%     fUpdateA = @(A) [A(:,1:end-1) sparse(size(A,1), nVar0+1-size(A,2)) A(:,end)];
%     cones = arrayfun(@(i) setfield(cones(i), 'A', fUpdateA(cones(i).A)), 1:numel(cones));
    qobj = [];
end


%% convert rotated cone to regular cone, does not improve performance
% for i=1:numel(cones.RotatedCone)
%     if ~cones.RotatedCone(i), continue; end
%     
%     j = sum(cones.coneSizes(1:i-1));
%     cones.A(j+(1:2),:) = [1 1; 1 -1]/sqrt(2)*cones.A(j+(1:2),:);
% %     cones.A = [ cones.A(1:j,:)
% %                 [1 1; 1 0; 0 1]*cones.A(j+(1:2),:);
% %                 cones.A(j+3:end,:) ];
% %     cones.coneSizes(i) = cones.coneSizes(i)+1;
%     cones.RotatedCone(i) = false;    
% end

%%
cmd = ['minimize info echo(' num2str(option.info) ')'];


% [~, res] = mosekopt('symbcon echo(0)');
% symbcon = res.symbcon;

%% linear constraints
prob.c = c;
Aeq = sparse(0, nVar0);
beq = zeros(0, 1);

%% conic constraints
ncone = numel(cones);
prob.cones = struct('type', [], 'subptr', [], 'sub', []);

% do not reduce the redundant variables for dualization, because it hurts the performance, 
% and it can be done later, i.e. all lambda0 can be removed from the problem
removeRedundantVariables = ~option.dualize;
    
if ncone>0
    if removeRedundantVariables
        i = find( sum(cones.A, 2)==1 & sum(cones.A.^2, 2)==1 );
        i = i( cones.A(i,end)==0 );
        [~, j] = find( cones.A(i,:) );
    %     cones.A( sub2ind(size(cones.A), i, j) ) == 1;
        [~, ic] = unique(j);  % in case some variables are involved in multiple cones
        ij = [i(ic) j(ic)];
        
        nConeVars = sum(cones.coneSizes);
        prob.c = [prob.c zeros(1, nConeVars-size(ij,1))];

        prob.cones.subptr = cumsum([1 cones.coneSizes]);
        prob.cones.subptr = prob.cones.subptr(1:end-1);

        isNewVar = true(nConeVars, 1);
        isNewVar(ij(:,1)) = false;

        coneVarId = nVar0 + cumsum(isNewVar);
        coneVarId(ij(:,1)) = ij(:,2);

        prob.cones.sub = coneVarId;
        prob.cones.type = double( cones.RotatedCone );
        M = cones.A(isNewVar,:);
        beq = M(:,end);
        Aeq = [ -M(:, 1:end-1) speye(sum(isNewVar)) ];
    else
        numNewVars = cones.coneSizes;
        prob.c = [prob.c zeros(1, sum(numNewVars))];
        prob.cones.subptr = cumsum([1 numNewVars]);
        prob.cones.subptr = prob.cones.subptr(1:end-1);
        prob.cones.sub = nVar0+1:numel(prob.c);
        prob.cones.type = double( cones.RotatedCone );
        M = cones.A;
        beq = M(:,end);
%         myblkdiag = @(x) blkdiag(x{:});
%         Aeq = [ -M(:, 1:end-1) myblkdiag(arrayfun(@speye, numNewVars, 'UniformOutput', false)) ];
        Aeq = [ -M(:, 1:end-1) speye(sum(numNewVars)) ];
    end
end


%% quadratic energy
for i=1:numel(qobj)
    nVar = numel(prob.c);
    nNewVar = size(qobj(i).A, 1) + 2;

    prob.c(end+(1:nNewVar)) = [qobj(i).wt zeros(1, nNewVar-1)];
    Aeq(end+(1:nNewVar-1), [1:nVar0 nVar+(2:nNewVar)]) = [[sparse(1,nVar0); -qobj(i).A(:, 1:end-1)]     speye(nNewVar-1)];
    beq(end+(1:nNewVar-1), 1) = [1/2; qobj(i).A(:,end)];

    prob.cones.type(end+1) = symbcon.MSK_CT_RQUAD;
    prob.cones.subptr(end+1) = numel(prob.cones.sub) + 1;
    prob.cones.sub(end+(1:nNewVar)) = nVar+(1:nNewVar);
end

prob.a = [sparse(a) sparse(size(a,1), size(Aeq,2)-size(a,2)); Aeq];
prob.blc = [bl; beq];
prob.buc = [bu; beq];


%% dual form
if ~option.dualize % solve in primal form
    tic; [~, res] = mosekopt(cmd, prob, option.param); toc;
    x = res.sol.itr.xx;
    en = res.sol.itr.pobjval;
else % manual dualize
    % http://docs.mosek.com/7.0/capi/Conic_quadratic_optimization__1.html
    
    auxVar = find( any(Aeq(:, nVar0+1:end)) ) + nVar0;
    auxVar2 = find( ~any(Aeq(:, nVar0+1:end)) ) + nVar0;

    lambdaIds = {1:nVar0, auxVar, auxVar2};
    szLambdas = [nVar0 numel(auxVar) numel(auxVar2)];

    ne0 = size(a, 1);
    nVar = sum( szLambdas ) + ne0;
    
    M = Aeq(:, 1:nVar0);

    dualprob = [];
    dualprob.c = zeros(1, nVar);
    dualprob.c( [sum(szLambdas)+(1:ne0) lambdaIds{2}] ) = prob.blc';

    dualprob.a = sparse( nVar0*2+szLambdas(3), nVar );
    dualprob.a(1:nVar0, [lambdaIds{2} sum(szLambdas)+(1:ne0)]) = -[M' sparse(a)'];
    if removeRedundantVariables % disabled
        freeConeVars = setdiff(1:nVar0, prob.cones.sub);
        nonFreeConeVars = setdiff(1:nVar0, freeConeVars);
        dualprob.a(sub2ind(size(dualprob.a), nonFreeConeVars, nonFreeConeVars)) = 1;
    end
    dualprob.a(nVar0+(1:szLambdas(3)+nVar0), [lambdaIds{3} 1:nVar0]) = speye( szLambdas(3)+nVar0 );

    dualprob.blc = [ prob.c( [1:nVar0 lambdaIds{3}] )'; zeros(nVar0, 1) ];
    dualprob.buc = dualprob.blc;
    dualprob.cones = prob.cones;

%     assert( norm(bl-bu)<1e-8, 'inequality constraints not supported yet for dualization');
%     iieq = find( bu-bl>1e-8 & ~isinf(bu) );
    iieq = find( bu-bl>1e-8 );
    if ~isempty(iieq)
        nuc = numel(iieq);

        if all(isinf(bu(iieq)))
            % suc == 0 => y>=0 => -y <= 0
%             nieq = numel(iieq);
%             dualprob.a(end+(1:nieq), sum(szLambdas)+1:end) = sparse(1:nieq, iieq, 1, nieq, ne0);
            dualprob.blx = -inf*ones(nVar,1);
            dualprob.bux =  inf*ones(nVar,1);
            dualprob.bux(sum(szLambdas)+iieq) = 0;
        else
            assert(false, 'not supported yet!');
            dualprob.c(end+(1:nuc)) = bu(iieq)-bl(iieq);
            dualprob.a = [ dualprob.a sparse(size(dualprob.a,1), nuc);];
        end
    end
    
    if false && ~any(dualprob.c(1:nVar0)) % disabled because does not improve performance
        dualprob.a = dualprob.a(1:nVar0+szLambdas(3), nVar0+1:end);
        dualprob.blc = dualprob.blc(1:nVar0+szLambdas(3));
        dualprob.buc = dualprob.blc;
        
        dualprob.cones.sub = dualprob.cones.sub - nVar0;
        dualprob.blx = dualprob.blx(nVar0+1:end);
        dualprob.bux = dualprob.bux(nVar0+1:end);
        dualprob.c = dualprob.c(nVar0+1:end);
        
        tic; [~, res] = mosekopt(cmd, dualprob, option.param); toc;
        x = res.sol.itr.snx;
        x = [M; a]\[beq - x(lambdaIds{2}-nVar0); bl+res.sol.itr.sux(sum(szLambdas(2:3))+1:end)];
    else
        tic; [~, res] = mosekopt(cmd, dualprob, option.param); toc;
        x = res.sol.itr.snx;
%         x(1:nVar0) = [M; a]\[beq - x(lambdaIds{2}); bl];
        x = [M; a]\[beq - x(lambdaIds{2}); bl+res.sol.itr.sux(sum(szLambdas)+1:end)];
    end
    en = -res.sol.itr.pobjval;
end

if energyInSqrt
    en = en.^2;
end

en = [en; x(qEnergyVarId)];
x = x(1:nc0);
status = res.sol.itr.solsta;
