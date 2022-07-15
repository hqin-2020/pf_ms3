import numpy as np
import scipy as sp

def simulate(θ_true, T):
    
    Azo = θ_true[0]; Azz = θ_true[1]; Bz = θ_true[2]
    Aso = θ_true[3]; Ass = θ_true[4]; Bs = θ_true[5]
    
    Z01 = 0
    Z02 = Azo[1,0]/(1-Azz[1,1])
    S0 = sp.linalg.solve((np.eye(3) - Ass), Aso)

    Z = np.zeros((2,T+1)) 
    S = np.zeros((3,T+1)) 
    Z[:,[0]] = np.array([[Z01],[Z02]])
    S[:,[0]] = S0

    np.random.seed(0)
    Wz = np.random.multivariate_normal(np.zeros(2), np.eye(2), T+1).T
    np.random.seed(1)
    Ws = np.random.multivariate_normal(np.zeros(3), np.eye(3), T+1).T

    for t in range(T):
        Z[:,[t+1]] = Azo + Azz @ Z[:,[t]] + Bz @ Wz[:,[t+1]]
        S[:,[t+1]] = Aso + Ass @ S[:,[t]] + Bs @ Ws[:,[t+1]]

    D = np.ones((3,1)) @ Z[[0],:] + S
    
    return D, Z, S