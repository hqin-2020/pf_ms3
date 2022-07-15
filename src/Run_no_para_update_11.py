import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import multiprocessing
import pickle
import os

from PF_no_para_update import *

if __name__ == '__main__':

    workdir = os.path.dirname(os.getcwd())
    srcdir = os.getcwd()
    datadir = workdir + '/data/'
    outputdir = '/project2/lhansen/pf_ms3/'
    seed = 11

    obs_series = pd.read_csv(datadir + 'data.csv', delimiter=',',header = None)
    obs_series = np.array(obs_series)

    T = obs_series.shape[1]
    N = 10_000

    case = 'actual data, seed = ' + str(seed) + ', T = ' + str(T) + ', N = ' + str(N)
    try: 
        casedir = outputdir + case  + '/'
        os.mkdir(casedir)
    except:
        casedir = outputdir + case  + '/'

    D_0 = obs_series[:,[0]]
    Input_0 = [[D_0, seed+i] for i in range(N)]
    pool = multiprocessing.Pool()
    X_t_particle = pool.map(init, tqdm(Input_0))
    del(Input_0)
    with open(casedir + 'X_0.pkl', 'wb') as f:
        pickle.dump(X_t_particle, f)
    
    for t in tqdm(range(T-1)):
        print(t)
        D_t_next = obs_series[:,[t+1]]
        Input = [[D_t_next, X_t_particle[i], seed+t+i] for i in range(N)]
        del(D_t_next)
        del(X_t_particle)
        pool = multiprocessing.Pool()
        Output = pool.map(recursive, Input)
        del(Input)
        X_t_next_particle = [i[0] for i in Output]
        ν_t_next_particle = [i[1] for i in Output]   
        del(Output) 
        with open(casedir + 'X_' + str(t+1) + '.pkl', 'wb') as f:
            pickle.dump(X_t_next_particle, f)

        w_t_next = ν_t_next_particle/np.sum(ν_t_next_particle)
        del(ν_t_next_particle)

        try:
            count_all = sp.stats.multinomial.rvs(N, w_t_next)
        except:
            for i in range(w_t_next.shape[0]):
                if w_t_next[i]>(np.sum(w_t_next[:-1]) - 1):
                    w_t_next[i] = w_t_next[i] - (np.sum(w_t_next[:-1]) - 1)
                    break
            count_all = sp.stats.multinomial.rvs(N, w_t_next)
        
        with open(casedir + 'w_' + str(t+1) + '.pkl', 'wb') as f:
            pickle.dump(w_t_next, f)
        del(w_t_next)
        with open(casedir + 'count_' + str(t+1) + '.pkl', 'wb') as f:
            pickle.dump(count_all, f)
        
        X_t_particle = []       
        for i in range(N):
            if count_all[i] != 0:
                for n in range(count_all[i]):
                    X_t_particle.append(X_t_next_particle[i])
        del(count_all)        
        del(X_t_next_particle)
        