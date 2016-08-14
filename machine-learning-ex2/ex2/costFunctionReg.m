function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);

J = (-1 / m) * (y'*log(h) + (1-y')*log(1-h));
grad = (1/m) * X'*(h-y);

theta_sqrt = theta.^2;

theta_sqrt_sum = 0;
for i = 2:length(theta)
	theta_sqrt_sum = theta_sqrt_sum + theta_sqrt(i);
end

J_reg = ((lambda / (2*m)) * theta_sqrt_sum);
J = J + J_reg;

tmp = grad(1);
grad = grad + (lambda/m)*theta;
grad(1) = tmp; 


% =============================================================

end
