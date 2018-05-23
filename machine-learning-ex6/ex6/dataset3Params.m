function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% SVM Parameters
a = logspace(-2,2,20);
b = zeros(20,20);
% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
for i = 1:numel(a)
    for j = 1:numel(a)
        model= svmTrain(X, y, a(i), @(x1, x2) gaussianKernel(x1, x2, sqrt(a(j)))); 
        predictions = svmPredict(model, Xval);
        b(i,j) = mean(double(predictions ~= yval));
    end
end

[M,I] = min(b(:));
[I_row, I_col] = ind2sub(size(b),I);
C = a(I_row);
sigma = sqrt(a(I_col));




% =========================================================================

end
