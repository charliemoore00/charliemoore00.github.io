""" 
newton-optimizer.py

implements 
- LinearModel
- LogisticRegression which inherits from LinearModel
- GradientDescentOptimizer
- NewtonOptimizer

"""

import torch
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        ARGS: 
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        
        return X@self.w

    def predict(self, X):

        """
        Compute the predictions for each data point in the feature matrix X. 
        The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: 
                vector predictions in {0.0, 1.0}. 
                y_hat.size() = (n,)
        """
        s = self.score(X)
        return (s > 0).int()
        
        pass 



class LogisticRegression(LinearModel):
    
    def loss(self, X, y):

        """
        Computes the empirical risk L(w) using the logistic loss function

        ARGUMENTS:
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() = (n,). 
            The Possible labels for y are {0, 1}

        RETURNS:
            the empirical risk L(w)
        
        *  *  *  *  *  *  *  *  *  *  *  *

        Logistic Loss Equation (1): 
            L(w) = mean[-yi log(sigmoid(si)) - (1-yi)log(1-sigmoid(si))]
        """

        self.score(X)                   # call our score function to get the weights w
        scores = X @ self.w             # get our score vector s to use in the logistic loss function

        def sigmoid(z):                 # implementation of sigmoid function
            return 1/(1 + torch.exp(-z))
    

        # use the logistic loss function (1) to calculate L(w)!
        loss = torch.mean(-y * torch.log(sigmoid(scores)) - (1 - y) * torch.log(1 - sigmoid(scores)))
        return loss
    

    def grad(self, X, y):
        
        """
        Computes the gradient of the empirical risk L(w)

        ARGUMENTS: 
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

            y, torch.Tensor: 
                the target vector. y.size() == (n,).
                The possible labels for y are {0, 1}

        RETURNS:
            the gradient of the empirical risk L(w)
            
        """

        y_ = y[:, None]                 # convert tensor with shape (n,) to shape (n,1)

        self.score(X)                   # call our score function to get the weights w
        scores = X @ self.w             # get our score vector s to use in the logistic loss function
        scores = scores[:, None]
        def sigmoid(z):                 # implementation of sigmoid function
            return 1/(1 + torch.exp(-z))
        
        gradients = (sigmoid(scores) - y_) * X
        gradient = torch.mean(gradients, dim=0)
        return gradient
    

    def hessian(self, X, y):

        """
        Computes Hessian Matrix H(w)

        ARGS: 
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

            y, torch.Tensor: 
                the target vector. y.size() == (n,).
                The possible labels for y are {0, 1}

        RETURNS:
            hessian - matrix of second derivatives of the loss function

        Process:

        Create matrix D: the diagonal matrix with entries 
            dsubkk(w) = sigmoid(s subk)(1 - sigmoid(s subk))

        use the formula H(w) = X^T D(w) X

        """

        def sigmoid(z):                 # implementation of sigmoid function
            return 1/(1 + torch.exp(-z))
        
        scores = X @ self.w             # get our score vector s to use in the logistic loss function

        # calculate the diagonal matrix D
        D = torch.diag(sigmoid(scores)*(1-sigmoid(scores)))

        # calculate the Hessian formula using D
        hessian = X.T @ D @ X

        return hessian



class GradientDescentOptimizer():

    def __init__(self, model):
        self.model = model
        self.w_old = None   # self.w_old will store Wk-1 for gradient descent with momentum

    def step(self, X, y, alpha, beta):

        """
        Implements Gradient Descent with Momentum
        It performs an update of the weights w

        ARGUMENTS:
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector

            alpha

            beta

        RETURNS:
            Nothing
            Updates the weights self.model.w
            Use equation (2):
            Wk+1 <-- Wk - alpha(grad(Wk)) + beta(Wk - Wk-1)
        """

        gradient = self.model.grad(X, y)    # calculate the gradient
        w = self.model.w                    # create var for weights

        if self.w_old == None:
            self.w_old = w

        # use equation (2) to calculate new weights Wk+1
        new_weights = w - alpha*gradient + beta*(w - self.w_old)
        
        self.w_old = w.clone()              # use in next equation as Wk-1
        self.model.w = new_weights          # update new W



class NewtonOptimizer():

    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha):

        """
        1. Compute gradient
        2. Compute hessian matrix
        3. Once we know how to calculate the gradient and the Hessian, we repeat the update
            w <-- w - alpha H(w)^-1  grad(W)
            until convergence. Here, alpha > is a learning rate and H(w)^-1 is the matrix inverse of the Hessian matrix
        
        ARGS:
            X, torch.Tensor: 
                the feature matrix. X.size() == (n, p), where n is 
                the number of data points and p is the number of 
                features. This implementation always assumes that 
                the final column of X is a constant column of 1s.

            y, torch.Tensor: 
                the target vector

            alpha, float:
                learning rate alpha > 0

        """

        # calculate the gradient
        gradient = self.model.grad(X, y)

        # calculate the Hessian matrix
        hessian = self.model.hessian(X, y)
    
        hessian_inv = torch.linalg.inv(hessian)

        # calculate new weights using inverted Hessian matrix and gradient
        w = self.model.w - alpha * hessian_inv @ gradient

        # update new weights
        self.model.w = w

    

