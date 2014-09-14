function [mus, sigmas, priors] = q5_gmminit(X, K, labels)
% Initializes a GMM model, given an initial clustering.

% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of gaussians for the GMM
%  labels: [m x 1] vector, containing the labels that the Kmeans algorithm assigned to the data.
%                  Each label l is an element of {1 ... K}, and it is associated with 
%                  the l-th gaussian.
% 
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                      covariance matrix for the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.

[m,n] = size(X);
%priors
priors = zeros(1,K);
for i = 1:K
    priors(1,i) = size(find(labels == i),1) / m;
end

%mus
mus = zeros(n,K);
for i = 1:K
    mus(:,i) = mean(X(find(labels == i),:))';
end

%sigmas
sigmas = zeros(n,n,K);
for i = 1:K
    X_j = X(find(labels == i),:);
    s = zeros(n,n);
    for j = 1:size(X_j,1)
        s = s + (X_j(j,:)'-mus(:,i)) * (X_j(j,:)'-mus(:,i))';
    end
    sigmas(:,:,i) = s / size(X_j,1);
end

end

