# NOTES: 1e-4 weight decay, gamma=1
# round 1 @ lr=0.0001
# round 2 @ lr=0.00001

#!/usr/bin/env python
import sys, os
local_path = '../'
sys.path.append(os.path.join(local_path, 'code/modules'))
import numpy as np
# from rnn_utils_copyNbedyn import Paper_NN, train_model, analyze_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from odelibrary import VDP, my_solve_ivp
from pdb import set_trace as bp


ode = VDP()
settings= {'method': 'RK45'}

N_traj = 10
# solver settings
T = 100
dt = 0.01
T += N_traj*dt # extra buffer for last window
t_eval = dt*np.arange(0, int(T/dt))
t_span = [t_eval[0], t_eval[-1]]

sol_list = []
for j in range(N_traj):
    sol = my_solve_ivp( ode.get_inits(), ode.rhs, t_eval, t_span, settings)
    sol_list.append(sol.T)
    fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
    axs.plot(t_eval[:len(sol)], sol)
    plt.savefig(os.path.join(local_path,'data/plots/X_train_VDP_multi_traj_short_{}'.format(j)))
    plt.close()


X_train = np.array(sol_list).T


save_path = os.path.join(local_path,'data/X_train_VDP_multi_traj_short.npy')
np.save(save_path, X_train.T)


###
# solver settings
T = 1e2
dt = 0.01
t_eval = dt*np.arange(0, int(T/dt))
t_span = [t_eval[0], t_eval[-1]]
x_warmup = my_solve_ivp( ode.get_inits(), ode.rhs, t_eval, t_span, settings)

u0 = x_warmup[-1]
T = 1e3
dt = 0.01
t_eval = dt*np.arange(0, int(T/dt))
t_span = [t_eval[0], t_eval[-1]]
X_test = my_solve_ivp( u0, ode.rhs, t_eval, t_span, settings)
fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
axs.plot(t_eval[:len(X_test)], X_test)
plt.savefig(os.path.join(local_path,'data/plots/X_test_VDP'))
plt.close()

save_path = os.path.join(local_path,'data/X_test_VDP.npy')
np.save(save_path, X_test.T)
