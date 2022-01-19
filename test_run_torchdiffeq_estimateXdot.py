#!/usr/bin/env python
import sys, os
local_path = './'
sys.path.append(os.path.join(local_path, 'code/modules'))
import numpy as np
# from rnn_utils_copyNbedyn import Paper_NN, train_model, analyze_models
from rnn_utils_torchdiffeq import Paper_NN, train_model, analyze_models
from pdb import set_trace as bp


# load L63 data sampled at dt=0.01
dt=0.01
train_path = os.path.join(local_path,'data/X_train_L63_CHAOS.npy')
test_path = os.path.join(local_path,'data/X_test_L63_CHAOS.npy')

X_train = np.load(train_path).T
X_test  = np.load(test_path).T
X_val = X_test[:,:2000]

# X_train = X_train[:,1000:5000]
print('Train shape:',X_train.shape)
print('Test shape:', X_val.shape)

# create new RNN object
my_rnn = Paper_NN(dim_output=3,
                    dim_hidden=200,
                    train_length=X_train.shape[1],
                    dt=dt,
                    activation='gelu')

# train the RNN
print('Begin RNN training...')
train_model(my_rnn, X_train.T, X_val.T,
            use_true_xdot=False,
            epochs=5000,
            learning_rate=0.1,
            step_size=500,
            batch_size=1000,
            gamma=0.5,
            n_warmup=100,
            output_dir='output/torchdiffeq_estXdot_v1')

### Print results for different test initializations
# print('Printing evaluation plots at different initial conditions of test set...')
# analyze_models(x0=X_val[:,0], model=my_rnn, dt=dt)
#
# # use last state of long trajectory
# analyze_models(x0=X_val[:,-1], model=my_rnn, dt=dt)
#
# # use perturbed state
# analyze_models(x0=X_val[:,-1] + 0.01, model=my_rnn, dt=dt)
