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
from nn_simple_cuda import Paper_NN, train_model
from pdb import set_trace as bp
from hpc_utils import dict_to_file, addLoggingLevel
from git import Repo
from odelibrary2 import *

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', default="L96Meps-1", type=str)
parser.add_argument('--T_long', default=500, type=float)
parser.add_argument('--eval_time_limit', default=600, type=int) # (seconds) limit time of test solutions (abort if taking too long and continue training)
parser.add_argument('--short_run', default=0, type=int)
parser.add_argument('--cheat_normalization', default=0, type=int)
parser.add_argument('--obs_noise_sd', default=0, type=float)
parser.add_argument('--hpc', default=0, type=int)
parser.add_argument('--do_normalization', default=1, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--min_lr', default=0, type=float)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--max_grad_norm', default=0, type=float)
parser.add_argument('--shuffle', default=1, type=int)
parser.add_argument('--multi_traj', default=0, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--batch_size', default=1000, type=int)
parser.add_argument('--dim_hidden', default=500, type=int)
parser.add_argument('--activation', default='gelu', type=str)
parser.add_argument('--use_bilinear', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--dt', default=0.01, type=float)
parser.add_argument('--plot_interval', default=100, type=int)
parser.add_argument('--output_dir', default='default_output', type=str)
FLAGS = parser.parse_args()

if FLAGS.hpc:
    base_dir = '/groups/astuart/mlevine/contRNN/{}'.format(FLAGS.ds_name.lower())
else:
    base_dir = 'output'
output_dir = os.path.join(base_dir, FLAGS.output_dir)
os.makedirs(output_dir, exist_ok=True)
dict_to_file(mydict=FLAGS.__dict__, fname=os.path.join(output_dir,"settings.log"))

log_fname = os.path.join(output_dir,"logfile.log")
addLoggingLevel(levelName='EXTRA', levelNum=logging.DEBUG + 5)
logging.basicConfig(filename=log_fname, level=logging.EXTRA, format="%(message)s \t %(asctime)s")
logger = logging.getLogger()
logger.info('###### BEGIN EXPERIMENT  #########')

local_repo = Repo(search_parent_directories=True)
local_branch = local_repo.active_branch.name
sha = local_repo.head.object.hexsha
logger.info('Using git branch "{}" on commit "{}"'.format(local_branch, sha))

# load L63 data sampled at dt=0.01
# dt=0.01

if FLAGS.ds_name=='L63':
    if FLAGS.multi_traj:
        train_path = os.path.join(local_path,'data/X_train_L63_multi_traj_short.npy')
    else:
        train_path = os.path.join(local_path,'data/X_train_L63_longer.npy')
    long_path = os.path.join(local_path,'data/X_test_L63.npy')
    X_train = np.load(train_path)
    X_long  = np.load(long_path)
else:
    long_path = os.path.join(local_path,'data/X_train_{}_longer.npy').format(FLAGS.ds_name)
    X_long  = np.load(long_path)
    # X_long = X_long[:9,:-200000]
    if FLAGS.multi_traj:
        train_path = os.path.join(local_path,'data/X_train_{}_multi_traj.npy').format(FLAGS.ds_name)
        X_train = np.load(train_path)
        X_train = X_train[:,:9,:500]
    else:
        X_train = X_long

X_train = torch.from_numpy(X_train)

# compute rhs (using notation from )
ode = L96M(eps = 2**(-1))
f0 = ode.slow(X_train[:ode.K], 0)
rhs_train = ode.full(0, X_train)
ydot_train = rhs_train
ydot_train[:ode.K] -= f0

# print('!!!!!!!!!!WARNING--using TEST DATA for training because it is bigger for now......!!!!!!!!!!!!!!')
# X_train = X_test
# X_train = X_train[:,1000:5000]
logger.info('X_train shape: {}'.format(X_train.shape))
logger.info('ydot_train shape: {}'.format(ydot_train.shape))

# train the NN on ydot
x_input = X_train.T.float()
y_output = ydot_train.T.float()

# create new NN object
model = Paper_NN(ode=ode, **FLAGS.__dict__)

# try to load the pre-trained RNN
if FLAGS.gpu:
    device = 'cuda'
else:
    device = 'cpu'

try:
    model = torch.load(os.path.join(output_dir, 'rnn.pt'), map_location=torch.device(device))
    logger.info('Loaded pre-trained model.')
    pre_trained=True
except:
    pre_trained=False
    logger.info('First time training this model')

if FLAGS.gpu:
    model = model.cuda()

n_train = X_train.shape[1]
batch_size = FLAGS.batch_size

FLAGS.output_dir = output_dir
logger.info('N train ='.format(n_train))
logger.info('Begin RNN training...')
try:
    # for 2-layer, bs=1000 was good. for 4-layer, bs=100 is good
    train_model(model, x_input, y_output,
                logger=logger,
                **FLAGS.__dict__)
except Exception:
    logger.exception('Fatal error in main loop.')
