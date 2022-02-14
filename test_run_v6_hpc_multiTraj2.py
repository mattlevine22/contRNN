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
from torchdiffeq import odeint
from dynamical_models import L63_torch
from rnn_utils_torchdiffeq_ivan_bugfix_cuda import Paper_NN, train_model
from pdb import set_trace as bp
from hpc_utils import dict_to_file

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', default=0, type=int)
parser.add_argument('--do_normalization', default=0, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--torchdata', default=0, type=int)
parser.add_argument('--use_f0', default=1, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dim_hidden', default=50, type=int)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--use_bilinear', default=0, type=int)
parser.add_argument('--match_endpoints', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--window', default=100, type=int)
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--known_inits', default=1, type=int)
parser.add_argument('--dt', default=0.01, type=float)
parser.add_argument('--output_dir', default='default_output', type=str)
FLAGS = parser.parse_args()

if FLAGS.hpc:
    base_dir = '/groups/astuart/mlevine/contRNN/l63'
else:
    base_dir = 'output'
output_dir = os.path.join(base_dir, FLAGS.output_dir)
os.makedirs(output_dir, exist_ok=True)
dict_to_file(mydict=FLAGS.__dict__, fname=os.path.join(output_dir,"settings.log"))

log_fname = os.path.join(output_dir,"logfile.log")
logging.basicConfig(filename=log_fname, level=logging.INFO, format="%(message)s \t %(asctime)s")
logger = logging.getLogger()
logger.info('###### BEGIN EXPERIMENT  #########')

# load L63 data sampled at dt=0.01
# dt=0.01
train_path = os.path.join(local_path,'data/X_train_L63_multi_traj.npy')
test_path = os.path.join(local_path,'data/X_test_L63.npy')

X_train = np.load(train_path).T
X_test  = np.load(test_path).T
X_val = X_test

# print('!!!!!!!!!!WARNING--using TEST DATA for training because it is bigger for now......!!!!!!!!!!!!!!')
# X_train = X_test

# X_train = X_train[:,1000:5000]
logger.info('Train shape: {}'.format(X_train.shape))
logger.info('Test shape: {}'.format(X_val.shape))

# create new RNN object
my_rnn = Paper_NN(
                    logger=logger,
                    use_f0=FLAGS.use_f0,
                    infer_normalizers=False,
                    n_layers=FLAGS.n_layers,
                    use_bilinear=FLAGS.use_bilinear,
                    dim_x=1,
                    dim_y=2,
                    dim_output=3,
                    dim_hidden=FLAGS.dim_hidden,
                    activation=FLAGS.activation)

if FLAGS.gpu:
    my_rnn = my_rnn.cuda()

# try to load the pre-trained RNN
try:
    my_rnn = torch.load(os.path.join(output_dir, 'rnn.pt'))
    logger.info('Loaded pre-trained model.')
    pre_trained=True
except:
    pre_trained=False
    logger.info('First time training this model')




n_train = X_train.shape[1]
window = FLAGS.window
batch_size = FLAGS.batch_size

if FLAGS.torchdata:
    dt = 0.01
    logger.info('Re-writing data with torchdiffeq.odeint solver---should achieve near zero loss w/ perfect model now!')
    times = torch.FloatTensor(dt*np.arange(n_train))
    u0 = torch.FloatTensor(X_train[:,0])
    X_train = odeint(L63_torch, y0=u0.reshape(-1,1), t=times).squeeze().T.data.numpy().T

if X_train.ndim==2:
    X_train = X_train[:,:,None]
if X_val.ndim==2:
    X_val = X_val[:,:,None]

logger.info('N train ='.format(n_train))
logger.info('Begin RNN training...')
logger.warning('WARNING---SHUFFLE IS FALSE FOR NOW.')
train_model(my_rnn, X_train, X_val,
            logger=logger,
            dt_data=FLAGS.dt,
            match_endpoints=FLAGS.match_endpoints,
            shuffle_train_loader=False, #good run did False
            use_gpu=FLAGS.gpu,
            do_normalization=FLAGS.do_normalization,
            known_inits=FLAGS.known_inits,
            pre_trained=pre_trained,
            weight_decay=0,
            epochs=FLAGS.epochs,
            learning_rate=FLAGS.lr,
            gamma=1,
            step_size=50,
            batch_size=FLAGS.batch_size,
            window=FLAGS.window,
            output_dir=output_dir)
