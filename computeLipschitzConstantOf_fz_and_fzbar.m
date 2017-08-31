function [L_fz, L_fzbar] = computeLipschitzConstantOf_fz_and_fzbar(L, indexOfMaxL, v, Phi, Psy)
%   v - vertices of the original cage.
%   Phi - the DOF of discrete Cauchy transform - holomorphic part.
%   Psy - the DOF of discrete Cauchy transform - anti-holomorphic part.
%   L - mxn matrix of MOC (Lipschitz constants) of all basis functions on all segments.
%   indexOfMaxL - mx1 vector where each element holds the index of the basis function with largest L.

tic

Phi_1 = Phi(indexOfMaxL(:, 1), :);
Phi_2 = Phi(indexOfMaxL(:, 2), :);

v_1 = v(indexOfMaxL(:, 1), :);
v_2 = v(indexOfMaxL(:, 2), :);

c_Phi = (Phi_1 - Phi_2) ./ (v_2 - v_1);
d_Phi = -Phi_1 -c_Phi.*v_1;

L_fz = sum(L.*abs(bsxfun(@plus, d_Phi, Phi.') + bsxfun(@times, c_Phi, v.')), 2);

Psy_1 = Psy(indexOfMaxL(:, 1), :);
Psy_2 = Psy(indexOfMaxL(:, 2), :);

c_Psy = (Psy_1 - Psy_2) ./ (v_2 - v_1);
d_Psy = -Psy_1 -c_Psy.*v_1;

L_fzbar = sum(L.*abs(bsxfun(@plus, d_Psy, Psy.') + bsxfun(@times, c_Psy, v.')), 2);

fprintf('computeLipschitzConstantOf_fz_and_fzbar time: %.5f\n', toc);

end



% function [L_fz_gpu, L_fzbar_gpu] = computeLipschitzConstantOf_fz_and_fzbar(L_gpu, indexOfMaxL_gpu, v_gpu, Phi, Psy)
% %   v - vertices of the original cage.
% %   Phi - the DOF of discrete Cauchy transform - holomorphic part.
% %   Psy - the DOF of discrete Cauchy transform - anti-holomorphic part.
% %   L - mxn matrix of MOC (Lipschitz constants) of all basis functions on all segments.
% %   indexOfMaxL - mx1 vector where each element holds the index of the basis function with largest L.
% 
% tic
% 
% Phi_gpu = gpuArray(Phi);
% Psy_gpu = gpuArray(Psy);
% 
% Phi_1_gpu = Phi_gpu(indexOfMaxL_gpu(:, 1), :);
% Phi_2_gpu = Phi_gpu(indexOfMaxL_gpu(:, 2), :);
% 
% v_1_gpu = v_gpu(indexOfMaxL_gpu(:, 1), :);
% v_2_gpu = v_gpu(indexOfMaxL_gpu(:, 2), :);
% 
% c_Phi_gpu = (Phi_1_gpu - Phi_2_gpu) ./ (v_2_gpu - v_1_gpu);
% d_Phi_gpu = -Phi_1_gpu -c_Phi_gpu.*v_1_gpu;
% 
% M1_gpu = bsxfun(@plus, d_Phi_gpu, Phi_gpu.');
% M2_gpu = bsxfun(@times, c_Phi_gpu, v_gpu.');
% 
% M12_gpu = abs(M1_gpu + M2_gpu);
% M12_gpu = L_gpu.*M12_gpu;
% L_fz_gpu = sum(M12_gpu, 2);
% 
% Psy_1_gpu = Psy_gpu(indexOfMaxL_gpu(:, 1), :);
% Psy_2_gpu = Psy_gpu(indexOfMaxL_gpu(:, 2), :);
% 
% c_Psy_gpu = (Psy_1_gpu - Psy_2_gpu) ./ (v_2_gpu - v_1_gpu);
% d_Psy_gpu = -Psy_1_gpu -c_Psy_gpu.*v_1_gpu;
% 
% M1_gpu = bsxfun(@plus, d_Psy_gpu, Psy_gpu.');
% M2_gpu = bsxfun(@times, c_Psy_gpu, v_gpu.');
% M12_gpu = abs(M1_gpu + M2_gpu);
% M12_gpu = L_gpu.*M12_gpu;
% L_fzbar_gpu = sum(M12_gpu, 2);
% 
% % L_fz = gather(L_fz_gpu);
% % L_fzbar = gather(L_fzbar_gpu);
% 
% fprintf('computeLipschitzConstantOf_fz_and_fzbar time: %.5f\n', toc);
% 
% end




% 
% function [L_fz, L_fzbar] = computeLipschitzConstantOf_fz_and_fzbar(L, indexOfMaxL, v, Phi, Psy)
% %   v - vertices of the original cage.
% %   Phi - the DOF of discrete Cauchy transform - holomorphic part.
% %   Psy - the DOF of discrete Cauchy transform - anti-holomorphic part.
% %   L - mxn matrix of MOC (Lipschitz constants) of all basis functions on all segments.
% %   indexOfMaxL - mx1 vector where each element holds the index of the basis function with largest L.
% 
%     tic
% 
%     Phi_1 = Phi(indexOfMaxL(:, 1), :);
%     Phi_2 = Phi(indexOfMaxL(:, 2), :);
% 
%     v_1 = v(indexOfMaxL(:, 1), :);
%     v_2 = v(indexOfMaxL(:, 2), :);
% 
%     c_Phi = (Phi_1 - Phi_2) ./ (v_2 - v_1);
%     d_Phi = -Phi_1 -c_Phi.*v_1;
% 
%     L_fz = sum(L.*abs(bsxfun(@plus, d_Phi, Phi.') + bsxfun(@times, c_Phi, v.')), 2);
% 
%     Psy_1 = Psy(indexOfMaxL(:, 1), :);
%     Psy_2 = Psy(indexOfMaxL(:, 2), :);
% 
%     c_Psy = (Psy_1 - Psy_2) ./ (v_2 - v_1);
%     d_Psy = -Psy_1 -c_Psy.*v_1;
% 
%     L_fzbar = sum(L.*abs(bsxfun(@plus, d_Psy, Psy.') + bsxfun(@times, c_Psy, v.')), 2);
% 
%     fprintf('computeLipschitzConstantOf_fz_and_fzbar time: %.5f\n', toc);
% 
% end

