function [prob_c, free_energy_e, likelihood_e] = q5_GM_Expectation(X, mus, sigmas, priors)
% Executes the Expectation-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  mus: [n x K] matrix containing the n-dimensional means of the K Gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%
% OUTPUT:
%  prob_c: [K x m] matrix, containing the posterior probabilities over the K Gaussians for the m examples.
%          Specifically, prob_c(j, i) represents the probability that the
%          i-th example belongs to the j-th Gaussian, 
%          i.e., P(z^(i) = j | X^(i,:))
%  free_energy_e: [1 x 1] scalar value representing the free energy value
%  likelihood_e: [1 x 1] scalar value representing the log-likelihood value

[m,n] = size(X);
K = size(mus,2);

%prob_c
prob_c = zeros(K, m);
prob_x = zeros(1, m);
for i = 1:m
    for j = 1:K
        prob_c(j,i) = priors(1,j) * exp(q5_logprobgauss(X(i,:), mus(:,j), sigmas(:,:,j)));
    end
    prob_x(1,i) = sum(prob_c(:,i));
    for j = 1:K
        prob_c(j,i) = prob_c(j,i) / prob_x(1,i);
    end
end

%free_energy_e
free_energy_e = 0;
for i = 1:m
    for j = 1:K
        free_energy_e = free_energy_e + prob_c(j,i)*log(prob_x(1,i));
    end
end

%likelihood_e
likelihood_e = sum(log(prob_x));


end