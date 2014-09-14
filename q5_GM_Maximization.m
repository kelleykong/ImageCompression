function [mus, sigmas, priors, free_energy_m, likelihood_m] = q5_GM_Maximization(X, prob_c)
% Executes Maximization-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  prob_c: [K x m] matrix, containing the the posterior probabilities over the K Gaussians for the m examples.
%                  Please see the comments in q5_GM_Expectation.m
%
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian.
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%  free_energy_m: [1 x 1] scalar value representing the free energy value
%  likelihood_m: [1 x 1] scalar value representing the log-likelihood value

[m, n] = size(X);
[K, ~] = size(prob_c);

mus = zeros(n, K);
priors = zeros(1, K);
sigmas = zeros(n, n, K);

free_energy_m = 0;
likelihood_m = 0;


s = zeros(K, 1);
for i = 1 : K
	s(i) = sum(prob_c(i, :));

	priors(i) = s(i) / m;
	mus(:, i) = X' * prob_c(i, :)' / s(i);
end

for i = 1 : K
	for j = 1 : m
		sigmas(:, :, i) = sigmas(:, :, i) + prob_c(i, j) * (X(j, :)' - mus(:, i)) * (X(j, :)' - mus(:, i))';
	end

	sigmas(:, :, i) = sigmas(:, :, i) / s(i);

end


for i = 1 : m
	joint_prob = 1 : K;
	for j = 1 : K
		joint_prob(j) = priors(j) * exp(q5_logprobgauss(X(i, :), mus(:, j), sigmas(:, :, j)));
	end
	margin_prob = sum(joint_prob);

	for j = 1 : K
		delta = prob_c(j, i) * log(joint_prob(j)) - prob_c(j, i) * log(prob_c(j, i));
        if ~isnan(delta) && ~isinf(delta)
		free_energy_m = free_energy_m + delta;
        end

	end
	likelihood_m = likelihood_m + log(margin_prob);
end

end
