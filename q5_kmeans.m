function [labels, means, distortions] = q5_kmeans(X, K, seeds_idx)
% Executes Kmeans clustering algorithm, using euclidean distances.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids.
% 
% OUTPUT:
%  labels: [m x 1] vector, containing the labels that the K-means algorithm assigned to the examples.
%                  labels(i) is an element of {1 ... K}, and it indicates the cluster ID associated to the i-th example
%  means: [n x K] matrix, containing the n-dimensional centroids of the K clusters.
%  distortions: [1 x num_iterations] vector, each element containing the total distortion at a particular iteration, i.e.
%                                    the sum of the squared Euclidean distances between the examples
%                                    and their associated centroids.

[m,n] = size(X);
new_distortion = 1e10;
last_distortion = 2e10;
distortions = [];
means = X(seeds_idx,:);
while(last_distortion - new_distortion >= 1e-6)
   dist = q5_dist2(means, X);
   [val, labels] = min(dist);
   labels = labels';
   
   last_distortion = new_distortion;
   new_distortion = 0;

   for i = 1:m
       new_distortion = new_distortion + val(1,i);
   end
   distortions = [distortions new_distortion];
   
   for i = 1:K
       means(i,:) = mean(X(find(labels == i),:));
   end

end
means = means';



end
