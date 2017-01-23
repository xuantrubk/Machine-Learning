function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % Format: theta(j) := theta(j) - (alpha/m)*sum((h(x) - y)*x(j))
    %                                 ----------?------------------
    %         theta(j) := theta(j) -         theta_change(j)

    %============Calculate the theta_change==========================   
    % The size of X is mxn, the size of h(x)-y is mx1
    % => sum = X'*(h(x)-y)
    %
    % Calculate the vector theta_change 
    theta_change = (alpha/m)*(X'*((X*theta).-y));
    
    %update the theta data  
    theta = theta - theta_change;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
