clear;
close all;

%% data and required toolboxes

addpath(genpath('data'));
addpath(genpath('sgwt_toolbox'));

%%

% Number of datasets (avg results reported)
num_datasets = 4;

% Power of Laplacian
k = 8; % higher k leads to better estimate of the cut-off frequency

% compare the classification accuracies
labelled_percentage = 0.01:0.01:0.5;
num_points = length(labelled_percentage);

error_list = zeros(num_points, num_datasets);

for iter = 1:num_datasets

    %% data
    
    fprintf(['\n\nloading set' num2str(iter) '...\n\n']);
    load(['sub' num2str(iter) '.mat']);

    N = size(A,1);

    %% cells to store optimal sampling sets
    
    % We greedily select of batch of nodes to sample. Hence not necessary 
    % to start from scratch when a larger subset of nodes is to be sampled. 
    
    queries = cell(length(labelled_percentage),1);  
    
    %% computation to be done only once

    
    % compute the symmetric normalized Laplacian matrix
    d = sum(A,2);
    d(d~=0) = d.^(-1/2);
    Dinv = spdiags(d,0,N,N);
    Ln = speye(N) - Dinv*A*Dinv;
    clear Dinv;

    % make sure the Laplacian is symmetric
    Ln = 0.5*(Ln+Ln.');

    % higher power of Laplacian
    Ln_k = Ln;
    for i = 1:(k-1)
        Ln_k = Ln_k*Ln;
    end
    Ln_k = 0.5*(Ln_k+Ln_k.');
    
    %% Choosing optimal sampling sets of different sizes
    
    prev_queries = []; % sampling set chosen in previous iteration
    
    prev_nqueries = 0; % number of labels queried so far
    cur_nqueries = 0; % number of labels queried in current iteration
    
    error = zeros(num_points,1);

    for index_lp = 1:length(labelled_percentage)
        fprintf('\n\n*** fraction of data labelled = %f ***\n\n', labelled_percentage(index_lp))
        nqueries = round(labelled_percentage(index_lp) * N);

        cur_nqueries = nqueries - prev_nqueries;

        [prev_queries, error(index_lp)] = proposed_active_ssl_inc(cur_nqueries, mem_fn, Ln, k, Ln_k, prev_queries);
                                                                                                    
        fprintf('classification error (proposed) = %f \n\n', error(index_lp));
        queries{index_lp} = prev_queries;

        
        prev_nqueries = nqueries;
    end

    error_list(:,iter) = error;
    
end

%% plots

figure, errorbar(labelled_percentage, mean(1-error_list,2),...
                std(1-error_list,0,2),'-sb');
