function [mus, sigmas, priors, likelihood_e, free_energy_e, likelihood_m, free_energy_m] = q5_GaussianMixture(X, mus_init, sigmas_init, priors_init, num_iterations)
% Learn a Gaussian mixture model (GMM) by using the Expectation-Maximization (EM) algorithm.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  mus_init: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas_init: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian
%  priors_init: [1 x K] vector, containing the mixture priors of the K Gaussians.
%  num_iterations: [1 x 1] scalar value, indicating the number of EM iterations.
%
% OUTPUT:
%  mus: [n x K] matrix containing the d-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                      covariance matrix of the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%  likelihood_e: [num_iterations x 1] vector containing the likelihood after each E-step.
%  free_energy_e: [num_iterations x 1] vector containing the free energy after each E-step.
%  likelihood_m: [num_iterations x 1] vector containing the likelihood after each M-step.
%  free_energy_m: [num_iterations x 1] vector containing the free energy after each M-step.

mus = mus_init;
sigmas = sigmas_init;
priors = priors_init;
for num = 1:num_iterations
    [prob_c, free_energy_e, likelihood_e] = q5_GM_Expectation(X, mus, sigmas, priors);
    [mus, sigmas, priors, free_energy_m, likelihood_m] = q5_GM_Maximization(X, prob_c);
end
    
end
