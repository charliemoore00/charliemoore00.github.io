""" 
logistic.py

implements 
- LinearModel
- LogisticRegression which inherits from LinearModel

"""

import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

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
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = self.score(X)
        return (s > 0).int()
        
        pass 



class LogisticRegression(LinearModel):
    
    def loss(self, X, y):

        """
        Computes the empirical risk L(w) using the logistic loss function

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
            where n is the number of data points and p is the 
            number of features. This implementation always assumes
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() = (n,). 
            The Possible labels for y are {0, 1}

        RETURNS:
            the empirical risk L(w)
            equation:
            mean[-yi log(sigmoid(si)) - (1-yi)log(1-sigmoid(si))]
        """

        self.score(X)                   # call our score function to get the weights w
        scores = X @ self.w             # get our score vector s to use in the logistic loss function
        def sigmoid(z):                 # implementation of sigmoid function
            return 1/(1 + torch.exp(-z))

        # use the logistic loss function to calculate L(w)!
        loss = torch.mean(-y * torch.log(sigmoid(scores)) - (1 - y) * torch.log(1 - sigmoid(scores)))
        return loss
    

    def grad(self, X, y):
        
        """
        Computes the gradient of the empirical risk L(w)

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector. y.size() == (n,).
            The possible labels for y are {0, 1}

        RETURNS:
            the gradient of the empirical risk L(w)
            equation:
            change of L(w) = mean(sigmoid(si) - yi)xi
            
        """

        y_ = y[:, None]                 # convert tensor with shape (n,) to shape (n,1)

        self.score(X)                   # call our score function to get the weights w
        scores = X @ self.w             # get our score vector s to use in the logistic loss function
        scores = scores[:, None]
        def sigmoid(z):                 # implementation of sigmoid function
            return 1/(1 + torch.exp(-z))
        
        #print(y_.shape)
        #print((sigmoid(scores) - y_).shape)
        #print(X.shape)
        
        gradients = (sigmoid(scores) - y_) * X
        gradient = torch.mean(gradients, dim=0)
        return gradient



class GradientDescentOptimizer():
    def dostuff():
        print("doing stuff")
        

        


# TESTING
def classification_data(n_points = 300, noise = 0.2, p_dims = 2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    y = 1.0*y
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,p_dims))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)
    
    return X, y

X, y = classification_data(noise = 0.5)

LR = LogisticRegression()
s = LR.loss(X, y)
g = LR.grad(X, y)
print(s)
print(g)