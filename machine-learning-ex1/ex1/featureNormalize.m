function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2)); 		% mean, for each column of features.
sigma = zeros(1, size(X, 2));	% std, for each column of features.

m = length(mu);

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


for i = 1:m
	feature_column = X(:,i);
	mu(1, i) = mean(feature_column);
	mu(1, i) = mean(feature_column);
	
	sigma(1, i) = std(feature_column);
	sigma(1, i) = std(feature_column);

	X_norm(:,i) = (feature_column - mu(1, i)) / sigma(1,i);
end

mu
sigma

% ============================================================

end
