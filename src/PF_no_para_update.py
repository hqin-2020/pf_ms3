import numpy as np
import scipy as sp
import seaborn as sns
sns.set()
import pickle
import os

workdir = os.path.dirname(os.getcwd())
srcdir = os.getcwd()
datadir = workdir + '/data/'
# with open(datadir + 'MLE_estimate.pkl', 'rb') as f:
#     θ_true = pickle.load(f)

λ, η = 0.5, 0
b11, b22 = 1, 0.5

As11, As12, As13,      = 0.9, 0.0, 0.0
As21, As22, As23, Aso2 = 0.0, 0.8, 0.0, 0
As31, As32, As33, Aso3 = 0.0, 0.0, 0.7, 0

Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = 4, 0, 3, 0, 0, 2

θ_true = (λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33)


def init(Input):

    D0, seed = Input
    np.random.seed(seed)
    λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ_true
    ones = np.ones([3,1])
    Ass = np.array([[As11, As12, As13],\
                    [As21, As22, As23],\
                    [As31, As32, As33]])
    Aso = np.array([[0.0],\
                    [Aso2],\
                    [Aso3]])
    Bs =  np.array([[Bs11, 0,    0],\
                    [Bs21, Bs22, 0],\
                    [Bs31, Bs32, Bs33]])
    
    μs = sp.linalg.solve(np.eye(3) - Ass, Aso) 
    Σs = sp.linalg.solve_discrete_lyapunov(Ass, Bs@Bs.T)
    
    β = sp.linalg.solve(np.hstack([Σs@np.array([[1,1],[0,-1],[-1,0]]), ones]).T, np.array([[0,0,1]]).T)                                     
    γ1 = np.array([[1],[0],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ2 = np.array([[0],[1],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ3 = np.array([[0],[0],[1]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    Γ = np.hstack([γ1, γ2, γ3])
    
    Z01 = β.T@(D0 - μs)
    Σz01 = 0.0
    Z02 = η/(1-λ)
    Σz02 = b22**2/(1-λ**2)
    S0 = Γ.T@(D0 - μs) + μs
    Σs0 = (1/(ones.T@sp.linalg.inv(Σs)@ones))[0][0]
    
    μ0 = np.array([[Z01[0][0]],\
                   [Z02],\
                   [S0[0][0]],\
                   [S0[1][0]],\
                   [S0[2][0]]])
    Σ0 = np.array([[Σz01,0.0,    0.0,   0.0,   0.0],\
                   [0.0,   Σz02, 0.0,   0.0,   0.0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0],\
                   [0.0,   0.0,    Σs0, Σs0, Σs0]]) 

    X0 = sp.stats.multivariate_normal.rvs(μ0.flatten(), Σ0).reshape(-1,1)

    return X0

def recursive(Input):
    
    Dt_next, Xt, seed = Input
    np.random.seed(seed)

    λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ_true

    Azo = np.array([[0],[η]])
    Azz = np.array([[1, 1],[0, λ]])
    Bz = np.array([[b11, 0],[0, b22]])

    Ass = np.array([[As11, As12, As13],\
                    [As21, As22, As23],\
                    [As31, As32, As33]])
    Aso = np.array([[0.0],\
                    [Aso2],\
                    [Aso3]])
    Bs =  np.array([[Bs11, 0,    0],\
                    [Bs21, Bs22, 0],\
                    [Bs31, Bs32, Bs33]])

    Bz1 = Bz[[0],:]
    Zt = Xt[0:2,:]; Zt1 = Zt[0,0]; Zt2 = Zt[1,0]
    St = Xt[2:5,:]
    ones = np.ones([3,1])

    Φ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, ones@Bz1@Bz.T)
    Γ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, Bs@Bs.T)
    
    mean = np.vstack([Azo + Azz@Zt + Φ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St),\
                      Aso + Ass@St + Γ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St)])
    cov = np.vstack([np.hstack([Bz@Bz.T, np.zeros([2,3])]),\
                     np.hstack([np.zeros([3,2]), Bs@Bs.T])]) -\
          np.vstack([Φ.T, Γ.T]) @ (ones@Bz1@Bz1.T@ones.T+Bs@Bs.T)@np.hstack([Φ, Γ])

    Xt_next = sp.stats.multivariate_normal.rvs(mean.flatten(), cov).reshape(-1,1)
    
    des_mean = ones*Zt1 +ones*Zt2 + Aso + Ass@St
    des_cov = ones@Bz1@Bz1.T@ones.T+Bs@Bs.T
    
    density = sp.stats.multivariate_normal.pdf(Dt_next.flatten(), des_mean.flatten(), des_cov)
    
    return [Xt_next, density]