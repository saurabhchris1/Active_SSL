function [queries, error_opt] = proposed_active_ssl_inc(num_queries_to_add, mem_fn, Ln, k, Ln_k, prev_queries)
%   AUTHOR: Akshay Gadde, USC
%   This function does the following: it finds the optimal set of nodes to 
%   add to the given sampling set. Using the labels on this net sampling
%   set, it predicts the unknown labels and reports the classification
%   error along with the net sampling set.

% % %
% PARAMETER DESCRIPTION
% 
% INPUT
% num_queries_to_add: number of nodes to add to the given sampling set
% mem_fn:  ground truth membership functions of each class
% Ln: normalized Laplacian
% k: Power of Laplacian while computing cutoff, higher the order,
% greater the accuracy, but the complexity is also higher.
% Ln_k: kth power of the Laplacian 
% 
% OUTPUT
% queries: sampling set as a collection of node indices
% err_opt: classification error
% % %

%% options

num_iter = 100;
% warning('off','all');
N = size(Ln,1);

%% compute optimal sampling set and store its cutoff frequency 

S_opt_prev = false(N,1);
S_opt_prev(prev_queries) = true;

[S_opt, cutoff] = compute_opt_set_inc(Ln_k, k, num_queries_to_add, S_opt_prev);
queries = find(S_opt);

%% reconstruction using POCS


norm_val = zeros(num_iter,1); % used for checking convergence

% reconstruction using POCS

% approximate low pass filter using SGWT toolbox
filterlen = 10;
alpha = 8;
freq_range = [0 2];
g = @(x)(1./(1+exp(alpha*(x-cutoff))));
c = sgwt_cheby_coeff(g,filterlen,filterlen+1,freq_range);


% initialization
mem_fn_du = mem_fn;
mem_fn_du(~S_opt,:) = 0;
mem_fn_recon = sgwt_cheby_op(mem_fn_du,Ln,c,freq_range);

for iter = 1:num_iter % takes fewer iterations
    % projection on C1
    err_s = (mem_fn_du-mem_fn_recon); 
    err_s(~S_opt,:) = 0; % error on the known set
    
    % projection on C2
    mem_fn_temp = sgwt_cheby_op(mem_fn_recon + err_s,Ln,c,freq_range); % err on S approx LP
    
    norm_val(iter) = norm(mem_fn_temp-mem_fn_recon); % to check convergence
    if (iter > 1 && norm_val(iter) > norm_val(iter-1) ), break; end % avoid divergence
    mem_fn_recon = mem_fn_temp;
end
% predicted class labels
[~,f_recon] = max(mem_fn_recon,[],2);

% true class lables
[~,f] = max(mem_fn,[],2);

% reconstruction error
error_opt = sum(f(~S_opt)~=f_recon(~S_opt))/sum(~S_opt); % error for unknown labels only