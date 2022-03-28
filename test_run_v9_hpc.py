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
from hpc_utils import dict_to_file, addLoggingLevel
from git import Repo

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', default=0, type=int)
parser.add_argument('--ds_name', default="L63", type=str)
parser.add_argument('--T_long', default=5e3, type=float)
parser.add_argument('--eval_time_limit', default=600, type=int) # (seconds) limit time of test solutions (abort if taking too long and continue training)
parser.add_argument('--backprop_warmup', default=1, type=int)
parser.add_argument('--warmup_type', default='forcing', type=str)
parser.add_argument('--noisy_start', default=1, type=int)
parser.add_argument('--short_run', default=0, type=int)
parser.add_argument('--cheat_normalization', default=1, type=int)
parser.add_argument('--obs_noise_sd', default=0, type=float)
parser.add_argument('--dim_x', default=1, type=int)
parser.add_argument('--dim_y', default=2, type=int)
parser.add_argument('--hpc', default=0, type=int)
parser.add_argument('--do_normalization', default=0, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--min_lr', default=0, type=float)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--step_size', default=10, type=int) # for stepLR
parser.add_argument('--gamma', default=1, type=float) # for stepLR
parser.add_argument('--max_grad_norm', default=0, type=float)
parser.add_argument('--shuffle', default=0, type=int)
parser.add_argument('--multi_traj', default=0, type=int)
parser.add_argument('--learn_inits_only', default=0, type=int)
parser.add_argument('--infer_ic', default=1, type=int)
# parser.add_argument('--torchdata', default=0, type=int)
parser.add_argument('--use_f0', default=0, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dim_hidden', default=50, type=int)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--use_bilinear', default=0, type=int)
parser.add_argument('--lambda_endpoints', default=0, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--window', default=100, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--epochs', default=10000, type=int)
parser.add_argument('--known_inits', default=1, type=int)
parser.add_argument('--dt', default=0.01, type=float)
parser.add_argument('--plot_interval', default=1000, type=int)
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
    X_long = X_long[:9]
    if FLAGS.multi_traj:
        train_path = os.path.join(local_path,'data/X_train_{}_multi_traj.npy').format(FLAGS.ds_name)
        X_train = np.load(train_path)
        # X_train = X_train[:,:9,:500]
        X_train = X_train[:,:9]
    else:
        X_train = X_long

# print('!!!!!!!!!!WARNING--using TEST DATA for training because it is bigger for now......!!!!!!!!!!!!!!')
# X_train = X_test
# X_train = X_train[:,1000:5000]
logger.info('Train shape: {}'.format(X_train.shape))
logger.info('Test shape: {}'.format(X_long.shape))

# create new RNN object
my_rnn = Paper_NN(
                    logger=logger,
                    adjoint=FLAGS.adjoint,
                    ds_name=FLAGS.ds_name,
                    gpu=FLAGS.gpu,
                    warmup_type=FLAGS.warmup_type,
                    use_f0=FLAGS.use_f0,
                    infer_normalizers=False,
                    infer_ic=FLAGS.infer_ic,
                    n_layers=FLAGS.n_layers,
                    use_bilinear=FLAGS.use_bilinear,
                    dim_x=FLAGS.dim_x,
                    dim_y=FLAGS.dim_y,
                    dim_hidden=FLAGS.dim_hidden,
                    activation=FLAGS.activation)

if FLAGS.gpu:
    my_rnn = my_rnn.cuda()

# try to load the pre-trained RNN
try:
    if FLAGS.gpu:
        my_rnn = torch.load(os.path.join(output_dir, 'rnn.pt'))
    else:
        my_rnn = torch.load(os.path.join(output_dir, 'rnn.pt'), map_location=torch.device('cpu'))
    logger.info('Loaded pre-trained model.')
    pre_trained=True
except:
    pre_trained=False
    logger.info('First time training this model')

n_train = X_train.shape[1]
window = FLAGS.window
batch_size = FLAGS.batch_size

FLAGS.output_dir = output_dir
# if FLAGS.torchdata:
#     dt = 0.01
#     logger.info('Re-writing data with torchdiffeq.odeint solver---should achieve near zero loss w/ perfect model now!')
#     times = torch.FloatTensor(dt*np.arange(n_train))
#     u0 = torch.FloatTensor(X_train[:,0])
#     X_train = odeint(L63_torch, y0=u0.reshape(-1,1), t=times).squeeze().T.data.numpy()

logger.info('N train ='.format(n_train))
logger.info('Begin RNN training...')
try:
    train_model(my_rnn, X_train.T, X_long.T,
                logger=logger,
                **FLAGS.__dict__)
except Exception:
    logger.exception('Fatal error in main loop.')
