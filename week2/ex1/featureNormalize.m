function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%  

% Calculate the mean value of X     
mu = mean(X); 
% Calculate the standard derivative value of X
sigma = std (X);
 
% Create matrices of mean value and standard derivative value
% for calculating the X_norm by created matrices 
 m = size(X,1);
 mu_matrix = ones(m,1)*mu;
 sigma_matrix = ones(m,1)*sigma;
  
  
% Return the X_norm value
  X_norm = (X-mu_matrix)./sigma_matrix;
% ============================================================

end
