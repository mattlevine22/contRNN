# NOTES: 1e-4 weight decay, gamma=1
# round 1 @ lr=0.0001
# round 2 @ lr=0.00001

#!/usr/bin/env python
import sys, os
local_path = './'
sys.path.append(os.path.join(local_path, 'code/modules'))
import numpy as np
# from rnn_utils_copyNbedyn import Paper_NN, train_model, analyze_models
import torch
from odelibrary import L63, my_solve_ivp
from pdb import set_trace as bp


ode = L63()
settings= {'method': 'RK45'}

N_traj = 100
# solver settings
T = 1e2
dt = 0.01
t_eval = dt*np.arange(0, int(T/dt))
t_span = [t_eval[0], t_eval[-1]]

sol_list = []
for _ in range(N_traj):
    sol = my_solve_ivp( ode.get_inits(), ode.rhs, t_eval, t_span, settings)
    sol_list.append(sol.T)

X_train = np.array(sol_list).T


save_path = os.path.join(local_path,'data/X_train_L63_multi_traj.npy')
np.save(save_path, X_train.T)
