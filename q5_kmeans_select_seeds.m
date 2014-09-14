function seeds_idx = q5_kmeans_select_seeds(X, K, mode)
% Returns an initial set of centroids (i.e. a set of seeds) for the Kmeans algorithm. 
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  mode: string, indicating the type of initilization. It can be either 'random' or 'diverse_set'.
% 
% OUTPUT:
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids; seeds_idx(i)
%                     should be an integer number between 1 and m.

X = X';
[n, m] = size(X);
if strcmp(mode, 'random')
    % random initialization
    seeds_idx = randperm(m);
    seeds_idx = seeds_idx(1:K);
elseif strcmp(mode, 'diverse_set')
    X = X';
    seeds_idx = zeros(1,K);
    seeds_idx(1,1) = 1;
    i = 1;
    while(i < K) 
        dist = q5_dist2(X(seeds_idx(1:i),:),X);
        if (i == 1)
            [min_dist loc] = max(dist);
        else
            [min_dist loc] = min(dist);
            [min_dist loc] = max(min(dist));
        end
        seeds_idx(1,i+1) = loc;
        i = i+1;
    end   

else
  error('parameter mode not recognized');
end

end