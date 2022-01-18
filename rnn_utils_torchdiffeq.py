#!/usr/bin/env python
import sys, os
local_path = './'
sys.path.append(os.path.join(local_path, 'code/modules'))
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.linalg import pinv
import torch
from odelibrary        import *
from torchdiffeq import odeint_adjoint, odeint
from dynamical_models import L63_torch
from timeit import default_timer
from pdb import set_trace as bp

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class TrivialNormalizer(object):
    '''Does not apply any transformation to inputs'''
    def __init__(self, x=None):
        super(TrivialNormalizer, self).__init__()

    def encode(self, x):
        return x

    def decode(self, x, sample_idx=None):
        return x

    def cpu(self):
        return

    def cuda(self):
        return


class Paper_NN(torch.nn.Module):
            def __init__(self,
                        train_length,
                        dt,
                        dim_x=1,
                        dim_y=2,
                        dim_output=3,
                        dim_hidden=100,
                        activation='gelu',
                        x_normalizer=TrivialNormalizer(),
                        m_normalizer=TrivialNormalizer()):
                super(Paper_NN, self).__init__()

                self.train_length = train_length # length of training trajectory
                self.dt           = dt # data time step and integration time step

                # assign parameter dimensions
                self.dim_x = dim_x
                self.dim_y = dim_y
                self.dim_hidden = dim_hidden
                self.dim_output = dim_output

                # assign normalizers
                self.x_normalizer = x_normalizer
                self.m_normalizer = m_normalizer

                # create model parameters and functions
                activation_dict = {
                    'relu': torch.nn.ReLU(),
                    'gelu': torch.nn.GELU(),
                    'tanh': torch.nn.Tanh()
                }
                self.activation                = activation_dict[activation]
                self.linearCell_in             = torch.nn.Linear(self.dim_x + self.dim_y, self.dim_hidden, bias=True)
                self.linearCell_out            = torch.nn.Linear(self.dim_hidden, self.dim_output, bias=True)

                y_latent = np.random.uniform(low=-0.5, high=0.5, size=(self.train_length, self.dim_y))
                self.y_latent = torch.nn.Parameter(torch.from_numpy(y_latent).float())


            def f_nn(self, inp):
                return self.linearCell_out(self.activation(self.linearCell_in(inp)))

            def f0(self, inp, sigma=10):
                '''takes full [observed, hidden] state vector and returns [observed,hidden] vector.
                    inputs and outputs are in original coordinates.
                    input dimensions: N x state_dims'''
                foo = torch.zeros_like(inp)
                foo[:,0] = -sigma*inp[:,0]
                return foo

            def rhs_numpy(self, inp, t=0):
                inp = torch.FloatTensor(inp.reshape(1,-1))
                foo = self.rhs(inp).detach().numpy() # not sure why gradient is still being computed here...it is called within model.eval() in testing loop.
                return foo

            def rhs_old(self, inp, t=0):
                '''input dimensions: N x state_dims'''

                # compute f0
                f0 = self.f0(inp) # observed dimensions of inp must be in original coordinates

                # get nn inputs (normalize observed input dimensions)
                nn_inp = inp.clone()
                nn_inp[:,:self.dim_x] = self.x_normalizer.encode(nn_inp[:,:self.dim_x])

                # get nn outputs (unnormalize observed output dimensions)
                nn_pred = self.f_nn(nn_inp)
                nn_pred[:,:self.dim_x] = self.m_normalizer.decode(nn_pred[:,:self.dim_x])

                foo = f0 + nn_pred
                return foo

            def rhs(self, inp, t=0):
                '''input dimensions: N x state_dims'''
                return self.f0(inp) + self.f_nn(inp)

            def forward(self, t, x, use_normalizers=False):
                '''input dimensions: N x state_dims'''
                if use_normalizers:
                    out = self.rhs_old(x)
                else:
                    out = self.rhs(x)
                return out


def batched_windows(T_total, T_window, shuffle=False):

    N_batches = int(np.floor(T_total / T_window)) - 1
    low = 0
    high = T_window

    # initialize index list
    inds      = np.zeros((N_batches,2)) # [start, end]
    ind_seed  = np.random.randint(low=low, high=high)
    inds[0,0] = ind_seed
    inds[0,1] = ind_seed + T_window
    # propagate indices
    for k in range(1, N_batches):
        inds[k,0] = inds[k-1,1]
        inds[k,1] = inds[k,0] + T_window

    if shuffle:
        np.random.shuffle(inds)

    return inds.astype(int)


def train_model(model,
                x_train,
                X_validation,
                do_batching=False,
                do_normalization=False,
                output_dir='output',
                obs_inds=[0],
                dt_data=0.01,
                use_true_xdot=False,
                rhs_true=L63_torch,
                epochs=500,
                batch_size=1000,
                n_warmup=100,
                learning_rate=0.1,
                weight_decay=0, #1e-4,
                step_size=100,
                gamma=0.5
            ):

    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    # set datasets as torch tensors
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(X_validation)

    # get datasizes
    ntrain = x_train.shape[0]
    ntest = X_validation.shape[0]

    # get xdot
    xdot_train_true = rhs_true(x_train.T, t=0).T
    xdot_test_true = rhs_true(x_test.T, t=0).T
    if use_true_xdot:
        xdot_train = xdot_train_true
        xdot_test = xdot_test_true
    else:
        xdot_train = torch.FloatTensor(np.gradient(x_train.data.numpy(), axis=0)/dt_data) # N x 3
        xdot_test = torch.FloatTensor(np.gradient(x_test.data.numpy(),axis=0)/dt_data) # N x 3
        err = torch.mean( (xdot_train_true - xdot_train)**2 ).item()
        print('True vs approx Xdot MSE = ', err)

    # get residual
    m_train = xdot_train# - model.f0(x_train).detach()
    m_test = xdot_test# - model.f0(x_test).detach()

    ### set normalizations
    # use this for inputing x into the NN
    if do_normalization:
        x_normalizer = UnitGaussianNormalizer(x_train[:, obs_inds])
        m_normalizer = UnitGaussianNormalizer(m_train[:, obs_inds])
    else:
        x_normalizer = TrivialNormalizer(x_train[:, obs_inds])
        m_normalizer = TrivialNormalizer(m_train[:, obs_inds])

    model.x_normalizer = x_normalizer
    model.m_normalizer = m_normalizer

    ## set up training data (in observed dimensions)
    x_train_now    = x_train[:-1, obs_inds] # x_k
    x_train_next   = x_train[1:, obs_inds]  # x_{k+1}
    m_train_now    = m_train[:-1, obs_inds] # m_k := xdot_k - f0(x_k)
    # (goal is to have inputs from x_train_now and predict m_train_now and x_train_next)

    ## set up testing data (in observed dimensions)
    x_test_now    = x_test[:-1, obs_inds] # x_k
    x_test_next   = x_test[1:, obs_inds]  # x_{k+1}
    m_test_now    = m_test[:-1, obs_inds] # m_k := xdot_k - f0(x_k)

    # create train/test loaders for batching
#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_normalized, ydot_train_normalized), batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_normalized, Y_validation), batch_size=batch_size, shuffle=False, drop_last=True)

    ## build optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # randomly chunk data into windows (batches)
    if do_batching:
        batch_inds_train = batched_windows(T_total=ntrain, T_window=batch_size, shuffle=False)
    else:
        batch_inds_train = [(0, ntrain-1)]
        batch_size = ntrain

    t_span_train = torch.FloatTensor([0, dt_data])
    myloss = torch.nn.MSELoss()
    # torch.autograd.set_detect_anomaly(True)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0

        b = -1
        for i_start, i_end in batch_inds_train:
            b+=1 # batch counter
            optimizer.zero_grad()

            # set up data
            x_now         = x_train_now[i_start:i_end]
            x_next        = x_train_next[i_start:i_end]
            m_now         = m_train_now[i_start:i_end]
            latent_now    = model.y_latent[i_start:i_end].clone() ## CHECK that this is immutable
            u_now = torch.hstack( (x_now, latent_now) ) # create full state vector

            # run forward model
            u_next_pred = odeint(model, y0=u_now, t=t_span_train)[-1]
            udot_now_pred = model.rhs(u_now)

            # compute losses
            m_loss        = myloss(udot_now_pred[:,:model.dim_x], m_now) #residual observed derivative
            # obs_next_loss = myloss(u_next_pred[:,:model.dim_x], x_next) #predicting next observed state (1-step RK45 predictor)
            # check that the rk45_loss = 0 in observed components unless there is noise in the data
            rk45_loss     = myloss(u_now[1:,model.dim_x:], u_next_pred[:-1, model.dim_x:]) # hidden states should not deviate much from RK45 solution

            # loss = m_loss + obs_next_loss + 100*rk45_loss
            loss = 0.1*m_loss + 8000*0.9*rk45_loss
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            if ep%10==0:
                K = u_next_pred.data.shape[-1]
                fig, axs = plt.subplots(nrows=K, figsize=(20, 10))
                for k in range(K):
                    axs[k].plot(u_next_pred[:,k].data, label='State {}'.format(k))
                    axs[k].legend()
                plt.savefig(os.path.join(output_dir,'TrainPlot_Batch{}_Epoch{}'.format(b,ep)))
                plt.close()

        train_loss /= len(batch_inds_train)
        if ep%10==0:
            print('Epoch', ep, 'Training---m_loss:', m_loss.item(), 'hidden rk45_loss:', rk45_loss.item())
            print('Epoch', ep, 'Train loss:', train_loss)

        scheduler.step()

        # validate by running off-data and predicting ahead
        model.eval()
        with torch.no_grad():
            u0 = u_next_pred[-1].data.numpy()

            if ep%10==0:
                print('solving for IC = ', u0)
                t_eval = dt_data*np.arange(0, 2000)
                t_span = [t_eval[0], t_eval[-1]]
                settings= {'method': 'RK45'}
                sol = my_solve_ivp( u0.reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)
                fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
                axs.plot(t_eval[:len(sol)], sol)
                plt.savefig(os.path.join(output_dir,'ContinueTraining_Short_Epoch{}'.format(ep)))
                plt.close()

            if ep%1000==0:
                t_eval = dt_data*np.arange(0, 1e6)
                t_span = [t_eval[0], t_eval[-1]]
                settings= {'method': 'RK45'}
                sol = my_solve_ivp( u0.reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)
                fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
                axs.plot(t_eval[:len(sol)], sol)
                plt.savefig(os.path.join(output_dir,'ContinueTraining_Long_Epoch{}'.format(ep)))
                plt.close()

                sns.kdeplot(sol[:,0], label='approx')
                sns.kdeplot(x_train[:,0], label='True')
                plt.legend()
                plt.title('L63 first coordinate KDE')
                plt.savefig(os.path.join(output_dir,'ContinueTraining_KDE_Epoch{}'.format(ep)))
                plt.close()

    return model

def plot_test(u, fig_path='fig_test'):
    ## Plot long-term trajectories
    K = u.shape[-1]
    for batch in range(u.shape[0]):
        fig, axs = plt.subplots(nrows=K, figsize=(20, 10))
        for k in range(K):
            axs[k].plot(u[batch,:,k], label='State {}'.format(k))
            axs[k].legend()
        plt.savefig(fig_path + '_batch{}'.format(batch))
        plt.close()


def analyze_models(x0, model, dt=0.01, true_ode=L63(), mdot_type='NN_4d'):

#     f_rnn = F_RNN(mdot_type=mdot_type)
    t_eval = dt*np.arange(0, 200000)
    t_span = [t_eval[0], t_eval[-1]]
    settings= {'method': 'RK45'}

    ## Generate solutions ##
    # solve approximate 3D ODE at initial condition x0
    print('Solving NN-approximate 3D ODE at initial condition x0=',x0)
    sol_4d_nn = my_solve_ivp( x0.reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)

    # solve true 3D ODE at initial condition x0
    print('Solving true 3D ODE at initial condition x0=',x0)
    # true_ode = L63()
    sol_3d_true = my_solve_ivp( x0.reshape(-1), lambda t, y: true_ode.rhs(y,t), t_eval, t_span, settings)


    print('Starting the plotting...')
    ## Plot short term trajectories

    n = int(10/dt)
    plt.figure()
    plt.plot(t_eval[:n],sol_3d_true[:n,0], label='L63 - 3D')
#     plt.plot(t_eval[:n],sol_4d_true[:n,0], label='L63 - 4D')
    plt.plot(t_eval[:n],sol_4d_nn[:n,0], label='NN - 3D')
    plt.title('L63 first coordinate (short-time)')
    plt.legend()

    ## Plot medium term trajectories
    n = int(250/dt)
    fig, axs = plt.subplots(nrows=3, figsize=(20, 10))
    for k in range(3):
        axs[k].plot(t_eval[:n],sol_3d_true[:n,k], label='True 3D')
#         axs[k].plot(t_eval[:n],sol_4d_true[:n,k], label='True 4D')
        axs[k].plot(t_eval[:n],sol_4d_nn[:n,k], label='NN 3D')
        axs[k].legend()
        axs[k].set_title('x_{}'.format(k))


    ## Plot long-term trajectories
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    n = int(1000/dt)
    axs[0].plot(t_eval[:n],sol_3d_true[:n,0], label='L63 - 3D')
#     axs[1].plot(t_eval[:n],sol_4d_true[:n,0], label='L63 - 4D')
    axs[1].plot(t_eval[:n],sol_4d_nn[:n,0], label='NN - 3D')
    for ax in axs:
        ax.legend()
    fig.suptitle('L63 first coordinate (long-time)')

    ## Plot invariant measure of trajectories for full available time-window
    n = len(sol_3d_true) #int(1000/dt)
    plt.figure()
    sns.kdeplot(sol_3d_true[:n,0], label='L63 - 3D')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    sns.kdeplot(sol_4d_nn[:n,0], label='NN - 3D')
    plt.title('L63 first coordinate KDE (all-time)')
    plt.legend()

    ## Plot invariant measure of trajectories for specific time-window (e.g. pre-collapse)
    n = int(100/dt)
    plt.figure()
    sns.kdeplot(sol_3d_true[:n,0], label='L63 - 3D')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    sns.kdeplot(sol_4d_nn[:n,0], label='NN - 3D')
    plt.title('L63 first coordinate KDE (pre-collapse, if any.)')
    plt.legend()
