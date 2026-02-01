import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo1(LSLROptimiser):
    """
    Gradient Descent for LSLR with optimal learning rate.
    
    Uses η = 1/L where L is the Lipschitz constant (largest eigenvalue of Hessian).
    For f(w) = (1/2n)||Xw - y||^2, Hessian = (1/n) X^T X
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
        
        ## TODO Use this for any pre-computations you need
        hessian = (2*(self.X).T @ self.X)/(self.X).shape[0]
        vals = np.linalg.eigvals(hessian)
        self.eta = 1/vals.max()



        ##



    def lr(self) -> float:
        ## TODO learning rate schedule
        # i think here we need to give lr as per the iteration
        return self.eta

    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        return params - self.lr()*self.stoch_grad(params, 0)

    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        return np.mean(((self.X @ w.reshape((-1, 1))).flatten()-(self.y).flatten())**2)

    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        n = (self.X).shape[0]
        return (2*(self.X).T @ self.X @ w.reshape((-1, 1)) - 2*(self.X).T @ self.y.reshape((-1, 1))).flatten()/n
    
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
       
        ## TODO Implement stochastic gradient computation
        n = (self.X).shape[0]
        num = np.random.choice(n)
        row = (self.X)[num, :].reshape((-1, 1))
        grad = row * ((row.T @ w - (self.y).reshape(-1, 1)[num, :]).reshape(1, ))
        return grad.flatten()

    def stoch_grad2(self, w: np.ndarray, gamma: int) -> np.ndarray:

        # mini batch sgd
        n = (self.X).shape[0]
        num = np.random.choice(n, size=(gamma, ), replace=False)
        subX = (self.X)[num, :]
        grad = ((subX).T @ subX @ w.reshape((-1, 1)) - (subX).T @ self.y.reshape((-1, 1))[num, :]).flatten()/gamma
        return grad.flatten()
    
    def coord_graddesc(self, w: np.ndarray, gamma: int) -> np.ndarray:
        gamma = np.random.choice(self.X.shape[1])
        e = np.zeros((self.X.shape[1], 1))
        e[gamma, 0] = (((self.X).T)[gamma, :] @ self.X @ w.reshape((-1, 1)) - ((self.X).T)[gamma, :] @ self.y.reshape((-1, 1))).flatten()/(self.X).shape[0]
        e = e*((self.X).shape[1])
        return e
    
    def weirdname(self, w: np.ndarray, gamma: int) -> np.ndarray:
        p_val = np.linalg.norm(self.X, 'fro')
        p_arr = np.linalg.norm(self.X, axis=0, ord ='fro') / p_val

        gamma = np.random.choice(self.X.shape[1], p=p_arr)
        e = np.zeros((self.X.shape[1], 1))
        e[gamma, 0] = (((self.X).T)[gamma, :] @ self.X @ w.reshape((-1, 1)) - ((self.X).T)[gamma, :] @ self.y.reshape((-1, 1))).flatten()/(self.X).shape[0]
        e = e / (p_arr[gamma])**2
        return e
    
    def minibatch_cord(self, w: np.ndarray, gamma: int) -> np.ndarray:
        subset = np.random.choice(self.X.shape[1], size = (gamma, ), replace=False)
        e = np.zeros((self.X.shape[1], 1))
        for col in subset:
            e[col, 0] = (((self.X).T)[gamma, :] @ self.X @ w.reshape((-1, 1)) - ((self.X).T)[gamma, :] @ self.y.reshape((-1, 1))).flatten()/(self.X).shape[0]
        
        e = e*(self.X.shape[1])/gamma

        return e
    
    def svrg_within_epoch(self, w_start: np.ndarray, w_now: np.ndarray, grad_start: np.ndarray) -> np.ndarray:
        # w given is the start of epoch 
        # we return grad after the end of epochs
        gamma = np.random.choice(self.X.shape[0])
        row = (self.X)[gamma, :].reshape((-1, 1))
        grad = row * ((row.T @ w_now - (self.y).reshape(-1, 1)[gamma, :]).reshape(1, ))
        grad_old = row*(((row.T @ w_start) - (self.y).reshape(-1, 1)[gamma, :]).reshape(1, ))
        return grad - grad_old + grad_start
    
    def svrg_epoch(self, w: np.ndarray, iters_per_epoch : int) -> np.ndarray:
        # returns w after an epoch
        w_start = w.copy()
        grad_start = self.full_grad(w_start)
        for iter in range(iters_per_epoch):
            w = w - self.lr() * self.svrg_within_epoch(w_start, w, grad_start)
        return w
    











    


        