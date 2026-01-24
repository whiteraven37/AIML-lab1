import numpy as np
from typing import Callable, Optional, Tuple, List, Any, Union

class func:
    def __init__(self):
        pass
    def __call__(self, x: np.ndarray) -> np.ndarray: # type: ignore
        return self.eval(x)
    def eval(self, x: np.ndarray) -> np.ndarray:# type: ignore
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass 


class LSLR(func):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        super().__init__()

    def eval(self, w: np.ndarray) -> float: # type: ignore
        ## TODO 
        return np.mean(((self.y).flatten() - (self.X @ w.reshape((-1, 1)).flatten()))**2)
        raise NotImplementedError("Function evaluation not implemented yet.")

    def grad(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO 
        return (2*(self.X).T @ self.X @ w - 2*(self.X).T @ self.y).flatten()/self.n_samples
        raise NotImplementedError("Gradient computation not implemented yet.")

    def hessian(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO
        return (2*(self.X).T @ self.X)/self.n_samples
        raise NotImplementedError("Hessian computation not implemented yet.")


class rosenbrock(func):
    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
       ## TODO: Implement the Rosenbrock function evaluation
        x1 = x[0]
        y1 = x[1]
        return (self.a - x1)**2 + self.b*(y1-x1**2)**2
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the Rosenbrock function gradient
        x1 = x[0]
        y1 = x[1]
        xgrad = -2*(self.a-x1)-4*self.b*x1*(y1-x1**2)
        ygrad = 2*self.b*(y1-x1**2)
        return np.array([xgrad, ygrad])

        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the Rosenbrock function Hessian
        x1 = x[0]
        y1 = x[1]
        el1 = [2-4*self.b*y1+12*self.b*x1*x1, -4*self.b*x1]
        el2 = [-4*self.b*x1, 2*self.b]
        return np.array([el1, el2])
        pass
class rot_anisotropic(func):
    def __init__(self, U: np.ndarray, V: np.ndarray, S: np.ndarray, b: np.ndarray) -> None:
        self.U = U
        self.V = V
        self.S = S
        self.b = b
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the rotated anisotropic function evaluation
        Q = self.U@self.S@(self.V).T
        x1 = x.reshape((-1, 1))
        val = (x1.T @ Q @ x1)[0, 0]
        oval = (self.b).T @ x1
        oval = oval[0, 0]
        return val-oval
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: 
        x1 = x.reshape((-1, 1))
        Q = self.U@self.S@(self.V).T
        m = Q + Q.T
        return ((m@x1)-self.b.reshape((-1, 1))).flatten()
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO:
        Q = self.U@self.S@(self.V).T
        m = Q + Q.T
        return m
        pass

if __name__ == "__main__":
    pass