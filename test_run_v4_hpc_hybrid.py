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
from rnn_utils_torchdiffeq_ivan_bugfix_cuda_hybrid import Paper_NN, train_model
from pdb import set_trace as bp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--known_inits', default=1, type=int)
parser.add_argument('--output_dir', default='test_givenIC_normalizeICs', type=str)
FLAGS = parser.parse_args()

if FLAGS.hpc:
    base_dir = '/groups/astuart/mlevine/contRNN/l63'
else:
    base_dir = 'output'
output_dir = os.path.join(base_dir, FLAGS.output_dir)

# load L63 data sampled at dt=0.01
dt=0.01
train_path = os.path.join(local_path,'data/X_train_L63_longer.npy')
test_path = os.path.join(local_path,'data/X_test_L63.npy')

X_train = np.load(train_path)
X_test  = np.load(test_path)
X_val = X_test

# print('!!!!!!!!!!WARNING--using TEST DATA for training because it is bigger for now......!!!!!!!!!!!!!!')
# X_train = X_test

# X_train = X_train[:,1000:5000]
print('Train shape:',X_train.shape)
print('Test shape:', X_val.shape)

# create new RNN object
my_rnn = Paper_NN(
                    infer_normalizers=False,
                    dim_x=1,
                    dim_y=2,
                    dim_output=3,
                    dim_hidden=50,
                    activation='gelu')

if FLAGS.gpu:
    my_rnn = my_rnn.cuda()

# try to load the pre-trained RNN
try:
    my_rnn = torch.load(os.path.join(output_dir, 'rnn.pt'))
    print('Loaded pre-trained model.')
    pre_trained=True
except:
    pre_trained=False
    print('First time training this model')


n_train = X_train.shape[1]
window = 100
batch_size = 100

print('N train =', n_train)
print('batch window =', window)
print('N batches =', batch_size)

print('Begin RNN training...')
train_model(my_rnn, X_train.T, X_val.T,
            shuffle_train_loader=True, #good run did False
            use_gpu=FLAGS.gpu,
            do_normalization=True,
            known_inits=FLAGS.known_inits,
            pre_trained=pre_trained,
            weight_decay=0,
            epochs=10000,
            learning_rate=1e-1,
            gamma=1,
            step_size=50,
            batch_size=batch_size,
            window=window,
            output_dir=output_dir)
