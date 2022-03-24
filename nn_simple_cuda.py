#!/usr/bin/env python
import sys, os
local_path = './'
sys.path.append(os.path.join(local_path, 'code/modules'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.linalg import pinv
from statsmodels.tsa.stattools import acf
import torch
from torch.utils.data import Dataset
from torchdiffeq import odeint_adjoint, odeint
from plotting_utils import plot_logs, train_plot
from dynamical_models import L63_torch
from timeit import default_timer
import logging
import signal
from contextlib import contextmanager
import pandas as pd
from pdb import set_trace as bp

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

## USAGE
# try:
#     with time_limit(10):
#         long_function_call()
# except TimeoutException as e:
#     print("Timed out!")


def myodeint(func, y0, t, adjoint=False):
    if adjoint:
        return odeint_adjoint(func, y0, t)
    else:
        return odeint(func, y0, t)

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
    def __init__(self, x):
        super(TrivialNormalizer, self).__init__()

    def encode(self, x):
        return x

    def decode(self, x, sample_idx=None):
        return x

    def cpu(self):
        return

    def cuda(self):
        return

def get_bs(x, batch_size, window):
    '''return a usable batch size for a dataset x given desired batch_size and window'''
    if x.ndim==3:
        N = x.shape[0] * x.shape[2]
    else:
        N = x.shape[0]
    min_bs = int(N / window)
    return min(batch_size, min_bs)


class Paper_NN(torch.nn.Module):
            def __init__(self,
                        ode=None,
                        n_layers=2,
                        use_bilinear=False,
                        dim_input=81,
                        dim_output=81,
                        dim_hidden=100,
                        activation='relu',
                        **kwargs):
                super(Paper_NN, self).__init__()

                if ode is None:
                    ode = L96M(eps = 2**(-1))
                self.ode = ode
                self.f0 = ode.slow
                self.dim_x = ode.K
                self.dim_y = ode.K * ode.J

                # assign parameter dimensions
                self.n_layers = n_layers
                self.use_bilinear = use_bilinear
                self.dim_input = dim_input
                self.dim_output = dim_output
                self.dim_hidden = dim_hidden

                # create model parameters and functions
                activation_dict = {
                    'relu': torch.nn.ReLU(),
                    'gelu': torch.nn.GELU(),
                    'tanh': torch.nn.Tanh()
                }
                self.activation_fun = activation_dict[activation]

                self.set_model()
                self.print_n_params()

            def set_model(self):
                if self.n_layers==1:
                    self.dim_hidden = self.dim_output

                self.activations = torch.nn.ModuleList([])
                self.linears = torch.nn.ModuleList([])
                self.bilinears = torch.nn.ModuleList([])

                for j in range(self.n_layers): #output of model is a trivial activation (for easy coding)
                    # first layer
                    if j==0:
                        self.activations.append(torch.nn.Identity())
                        self.linears.append(torch.nn.Linear(self.dim_input, self.dim_hidden, bias=True)) # first layer
                        if self.use_bilinear:
                            self.bilinears.append(torch.nn.Bilinear(self.dim_input, self.dim_input, self.dim_hidden, bias=False)) # first layer
                    elif j < (self.n_layers-1): # internal layers
                        self.activations.append(self.activation_fun)
                        self.linears.append(torch.nn.Linear(self.dim_hidden, self.dim_hidden, bias=True)) # interior layers 2...n-1
                        if self.use_bilinear:
                            self.bilinears.append(torch.nn.Bilinear(self.dim_hidden, self.dim_hidden, self.dim_hidden, bias=False)) # interior layers 2...n-1
                    else: # last layer
                        self.activations.append(self.activation_fun)
                        self.linears.append(torch.nn.Linear(self.dim_hidden, self.dim_output, bias=True))
                        if self.use_bilinear:
                            self.bilinears.append(torch.nn.Bilinear(self.dim_hidden, self.dim_hidden, self.dim_output, bias=False))

                return

            def compute_grad_norm(self):
                '''Compute the norm of the gradients per layer.'''
                grad_norm = {}
                for l in range(self.n_layers):
                    layer_norm = 0
                    layer_norm += self.linears[l].weight.grad.detach().data.norm(2).item()**2
                    layer_norm += self.linears[l].bias.grad.detach().data.norm(2).item()**2
                    if self.use_bilinear:
                        layer_norm += self.bilinears[l].weight.grad.detach().data.norm(2).item()**2
                        layer_norm += self.bilinears[l].bias.grad.detach().data.norm(2).item()**2
                    grad_norm[l] = layer_norm ** 0.5

                try:
                    grad_norm['y0'] = self.y0.grad.detach().data.norm(2).item()
                except:
                    pass
                return np.array(list(grad_norm.values()))

            def print_n_params(self):
                foo_all = sum(p.numel() for p in self.parameters())
                foo_tble = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print('{} total parameters ({} are trainable)'.format(foo_all, foo_tble))

            def f_nn(self, inp):
                for l in range(self.n_layers):
                    inp = self.activations[l](inp)
                    if self.use_bilinear:
                        inp = self.linears[l](inp) + self.bilinears[l](inp,inp) # easy way to cope with large initialzations of bilinear cells
                    else:
                        inp = self.linears[l](inp)
                return inp

            def rhs(self, t, x):
                '''Input and output in original coordinates'''
                nn_inp = self.x_normalizer.encode(x)
                foo = self.y_normalizer.decode(self.f_nn(nn_inp)).data.squeeze()
                foo[:self.dim_x] += self.f0(x[:self.dim_x], t) # use f0
                return foo


            def forward(self, x):
                '''input dimensions: N x state_dims'''
                return self.f_nn(x)

def train_model(model,
                x_input,
                y_output,
                short_run=False,
                logger=None,
                gpu=False,
                pre_trained=False,
                do_normalization=False,
                output_dir='output',
                obs_inds=None,
                dt=0.01,
                epochs=500,
                batch_size=3,
                lr=0.1,
                weight_decay=0, #1e-4, or 1e-5?
                min_lr=0,
                patience=10,
                max_grad_norm=0,
                shuffle=False,
                plot_interval=1000,
                T_long=5e2,
                eval_time_limit=600, # seconds
                **kwargs):

    if obs_inds is None:
        obs_inds = [k for k in range(model.dim_x)]

    fast_plot_interval = max(1, int(plot_interval / 10))
    # batch_size now refers to the number of windows selected in an epoch
    # window refers to the size of a given window

    sol_3d_true_kde = None

    # make output directory
    os.makedirs(output_dir, exist_ok=True)
    if logger is None:
        log_fname = os.path.join(output_dir,"logfile.log")
        logging.basicConfig(filename=log_fname, level=logging.INFO)
        logger = logging.getLogger()

    summary_dir = os.path.join(output_dir,'summary')
    os.makedirs(summary_dir, exist_ok=True)

    plot_dir = os.path.join(output_dir,'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # set datasets as torch tensors

    if do_normalization:
        logger.info('Doing normalization, go go!')
        x_normalizer = UnitGaussianNormalizer(x_input)
        y_normalizer = UnitGaussianNormalizer(y_output)
    else:
        x_normalizer = TrivialNormalizer(x_input)
        y_normalizer = TrivialNormalizer(y_output)

    # normalize data
    x_normalized = x_normalizer.encode(x_input)
    ydot_normalized = y_normalizer.encode(y_output)

    if gpu:
        x_normalizer.cuda()
        y_normalizer.cuda()
    model.x_normalizer = x_normalizer
    model.y_normalizer = y_normalizer



    # make dataset
    full_dataset = torch.utils.data.TensorDataset(x_normalized, ydot_normalized)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # create train/test loaders for batching
    bs_train = min(len(train_set), batch_size)
    bs_test = min(len(test_set), batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs_train, shuffle=shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs_test, shuffle=False, drop_last=True)
    logger.info('Effective batch_size: Training = {}, Testing = {}'.format(bs_train, bs_test))
    logger.info('Number of batches per epoch: Training = {}, Testing = {}'.format(len(train_loader), len(test_loader)))

    # create initial conditions for each window
    my_list = ['y0']
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    opt_param_list = [{'params': base_params, 'weight_decay': weight_decay, 'learning_rate': lr}]
    # optimizer = torch.optim.Adam([{'params': base_params, 'weight_decay': weight_decay}], lr=learning_rate, weight_decay=0)
    optimizer = torch.optim.Adam(opt_param_list, lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=epochs, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, min_lr=min_lr, patience=patience)

    lr_history = {key: [] for key in range(len(optimizer.param_groups))}
    train_loss_history = []
    test_loss_history = []
    grad_norm_pre_clip_history = []
    grad_norm_post_clip_history = []
    time_history = []
    myloss = torch.nn.MSELoss()
    t_outer = default_timer()
    for ep in range(epochs):
        t1 = default_timer()
        time_history.append((t1 - t_outer)/60/60)
        for grp in range(len(optimizer.param_groups)):
            lr_history[grp].append(optimizer.param_groups[grp]['lr'])
        model.train()
        test_loss = 0
        train_loss = 0
        grad_norm_pre_clip = 0
        grad_norm_post_clip = 0

        batch = -1
        for x, y in train_loader:
            if gpu:
                x, y = x.cuda(), y.cuda()

            batch += 1
            optimizer.zero_grad()

            out = model(x).reshape(batch_size, -1)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            # compute gradient norms for monitoring
            grad_norm_pre_clip += model.compute_grad_norm()

            # clip gradient norm
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            # compute gradient norms for monitoring
            grad_norm_post_clip += model.compute_grad_norm()

            optimizer.step()
            train_loss += loss.item()

        # regularized loss
        train_loss /= len(train_loader)
        train_loss_history += [train_loss]

        # grad norms
        grad_norm_pre_clip /= len(train_loader)
        grad_norm_pre_clip_history += [grad_norm_pre_clip]
        grad_norm_post_clip /= len(train_loader)
        grad_norm_post_clip_history += [grad_norm_post_clip]

        scheduler.step(train_loss)

        # validate by running off-data and predicting ahead
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                if gpu:
                    x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, -1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                test_loss += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()


            # regularized loss
            test_loss /= len(test_loader)
            test_loss_history += [test_loss]


            t2 = default_timer()
            if ep%fast_plot_interval==0:
                logger.info('Epoch {}, Train loss {}, Test loss {}, Time per epoch [sec] = {}'.format(ep, train_loss, test_loss, round(default_timer() - t1, 2)))
                torch.save(model, os.path.join(output_dir, 'rnn.pt'))
                plot_logs(x={'Time':time_history}, name=os.path.join(summary_dir,'timer'), title='Cumulative Training Time (hrs)', xlabel='Epochs')
                plot_logs(x={'Train':train_loss_history, 'Test':test_loss_history}, name=os.path.join(summary_dir,'loss_history'), title='Loss', xlabel='Epochs')
                plot_logs(x=lr_history, name=os.path.join(summary_dir,'learning_rates'), title='Learning Rate Schedule', xlabel='Epochs')
                gn_dict = {'Layer {}'.format(l): np.array(grad_norm_pre_clip_history)[:,l] for l in range(len(grad_norm_pre_clip))}
                plot_logs(x=gn_dict, name=os.path.join(summary_dir,'grad_norm_pre_clip'), title='Gradient Norms (Pre-Clip)', xlabel='Epochs')
                gn_dict = {'Layer {}'.format(l): np.array(grad_norm_post_clip_history)[:,l] for l in range(len(grad_norm_post_clip))}
                plot_logs(x=gn_dict, name=os.path.join(summary_dir,'grad_norm_post_clip'), title='Gradient Norms (Post-Clip)', xlabel='Epochs')

                if ep%(10*plot_interval)==0 and ep>0:
                    outdir = os.path.join(plot_dir, 'epoch{}'.format(ep))
                    Tl = T_long
                    evt = eval_time_limit
                elif ep%plot_interval==0:
                    outdir = os.path.join(plot_dir, 'afast_plots/epoch{}'.format(ep))
                    Tl = T_long/100
                    evt = int(eval_time_limit/10)

                t0_local = default_timer()
                x0 = x_input[0].squeeze()
                try:
                    with time_limit(evt):
                        sol_3d_true_kde = test_plots(x0=x0, logger=logger, sol_3d_true=x_input, sol_3d_true_kde=sol_3d_true_kde, rhs_nn=model.rhs, rhs_true=model.ode.full,  T_long=Tl, output_path=outdir, obs_inds=[k for k in range(model.dim_x)], gpu=gpu)
                except TimeoutException as e:
                    logger.info('Finished long-term model evaluation runs [TIMED OUT].')
                logger.extra('Test plots took {} seconds'.format(round(default_timer() - t0_local, 2)))

    # run final test plots
    for tl in [T_long, T_long*5]:
        outdir = os.path.join(plot_dir, 'final_Tlong{}'.format(tl))
        t0_local = default_timer()
        x0 = x_input[0].squeeze()
        test_plots(x0=x0, logger=logger, sol_3d_true=x_input, sol_3d_true_kde=sol_3d_true_kde, rhs_nn=model.rhs, rhs_true=model.ode.full,  T_long=tl, output_path=outdir, obs_inds=[k for k in range(model.dim_x)], gpu=gpu)
        logger.extra('FINAL Test plots took {} seconds for T_long = {}'.format(round(default_timer() - t0_local, 2), tl))

    return model


def test_plots(x0, logger, rhs_nn, nn_normalizer=None, sol_3d_true=None, sol_3d_true_kde=None, rhs_true=None, T_long=5e1, output_path='outputs', obs_inds=[0], gpu=False, Tacf=10, kde_subsample=500000):

    if sol_3d_true is None and rhs_true is None:
        raise('Must specify either a true RHS or a true solution')

    os.makedirs(output_path, exist_ok=True)

    # solver settings
    dt = 0.01
    t_eval = dt*np.arange(0, int(T_long/dt))
    t_span = [t_eval[0], t_eval[-1]]
    settings= {'method': 'RK45'}

    ## Generate solutions ##
    # solve true 3D ODE at initial condition x0
    if sol_3d_true is None:
#         sol_3d_true = my_solve_ivp( x0.reshape(-1), rhs_true, t_eval, t_span, settings)
        t0_local = default_timer()
        sol_3d_true = odeint(rhs_true, y0=x0, t=torch.Tensor(t_eval)).squeeze(1)
        logger.extra('Solving True-system for T_long={} took {} seconds'.format(T_long, round(default_timer() - t0_local, 2)))

    t0_local = default_timer()
    K = len(obs_inds)
    # solve approximate 3D ODE at initial condition x0
    if nn_normalizer is None:
        if gpu:
            sol_4d_nn = odeint(rhs_nn, y0=x0.reshape(-1).cuda(), t=torch.Tensor(t_eval).cuda()).squeeze(1)
        else:
            sol_4d_nn = odeint(rhs_nn, y0=x0.reshape(-1), t=torch.Tensor(t_eval)).squeeze(1)
#             sol_4d_nn = torch.FloatTensor(my_solve_ivp( x0.reshape(-1), rhs_nn, t_eval, t_span, settings))
    else:
        if gpu:
            sol_4d_nn = odeint(rhs_nn, y0=nn_normalizer.encode(x0).reshape(-1).cuda(), t=torch.Tensor(t_eval).cuda()).squeeze(1)
        else:
            sol_4d_nn = odeint(rhs_nn, y0=nn_normalizer.encode(x0).reshape(-1), t=torch.Tensor(t_eval)).squeeze(1)

        sol_4d_nn = nn_normalizer.decode(sol_4d_nn).cpu().data.numpy()
    logger.extra('Solving NN-system for T_long={} took {} seconds'.format(T_long, round(default_timer() - t0_local, 2)))

    nn_max = len(sol_4d_nn)
    true_max = len(sol_3d_true)

    len_dict = {'short': min(nn_max, int(10/dt)),
                    'medium': min(nn_max, int(250/dt)),
                    'long': nn_max}
    for key in len_dict:
        n = len_dict[key]
        fig, axs = plt.subplots(constrained_layout=True, nrows=K, figsize=(15, 15), sharex=True)
        for k in range(K):
            axs[k].plot(t_eval[:n],sol_3d_true[:n,k], label='True 3D')
    #         axs[k].plot(t_eval[:n],sol_4d_true[:n,k], label='True 4D')
            axs[k].plot(t_eval[:n],sol_4d_nn[:n,k], label='NN 3D')
            if k==0:
                axs[k].legend()
            axs[k].set_ylabel(r'$x_{}$'.format(k))
        axs[k].set_xlabel('Time')
        plt.savefig(os.path.join(output_path, 'trajectory_{}.pdf'.format(key)), format='pdf')
        plt.close()

    ## Plot combined invariant measure of all states
    ## Plot invariant measure of trajectories for full available time-window
    t0_local = default_timer()
    n = len(sol_3d_true) #int(1000/dt)
    n_burnin = int(0.1*n)
    fig, axs = plt.subplots(figsize=(20, 10))
    if sol_3d_true_kde is None:
        sol_3d_true_kde = sns.kdeplot(np.random.choice(sol_3d_true[n_burnin:,obs_inds].reshape(-1), size=kde_subsample), label='True system').get_lines()[0].get_data()
    else:
        axs.plot(sol_3d_true_kde[0], sol_3d_true_kde[1], label='True system')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    n_burnin = int(0.1*nn_max)
    sns.kdeplot(np.random.choice(sol_4d_nn[n_burnin:,obs_inds].reshape(-1), size=kde_subsample), label='NN system')
    plt.title('First coordinate KDE (all-time)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_stateAll_long.pdf'.format(k)), format='pdf')
    plt.close()

    ## Plot invariant measure of trajectories for specific time-window (e.g. pre-collapse)
    n = min(nn_max, int(100/dt))
    fig, axs = plt.subplots(figsize=(20, 10))
    axs.plot(sol_3d_true_kde[0], sol_3d_true_kde[1], label='True system')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    sns.kdeplot(np.random.choice(sol_4d_nn[:n,obs_inds].reshape(-1), size=kde_subsample), label='NN system')
    plt.title('First coordinate KDE (pre-collapse, if any.)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_stateAll_preCollapse.pdf'.format(k)), format='pdf')
    plt.close()
    logger.extra('Plotting invariant measures took {} seconds'.format(round(default_timer() - t0_local, 2)))

    ## Plot Autocorrelation Function
    t0_local = default_timer()
    n_burnin_approx = int(0.1*len(sol_4d_nn))
    n_burnin_true = int(0.1*len(sol_3d_true))
    Tacf = min(Tacf, dt*(len(sol_4d_nn) - n_burnin_approx)/2, dt*(len(sol_3d_true) - n_burnin_true)/2)
    nlags = int(Tacf/dt) - 1
    lag_vec = dt*np.arange(0,nlags+1)

    acf_error_all = []
    df = pd.DataFrame()
    for k in obs_inds:
        acf_approx = acf(sol_4d_nn[n_burnin_approx:,k], fft=True, nlags=nlags) #look at first component
        acf_true = acf(sol_3d_true[n_burnin_true:,k].data.numpy(), fft=True, nlags=nlags) #look at first component
        acf_error = np.mean((acf_true - acf_approx)**2)
        acf_error_all.append(acf_error)

        df_approx = pd.DataFrame({'Time Lag': lag_vec, 'ACF': acf_approx, 'Type': 'NN system', 'State': k})
        df_true = pd.DataFrame({'Time Lag': lag_vec, 'ACF': acf_true, 'Type': 'True system', 'State': k})
        df = pd.concat([df, df_approx, df_true], ignore_index=True)

        fig, axs = plt.subplots(figsize=(20, 10))
        axs.plot(lag_vec, acf_true, label='True system')
        axs.plot(lag_vec, acf_approx, label='NN system')
        axs.set_xlabel('Time Lag')
        plt.title('Autocorrelation Functions (component {})'.format(k))
        plt.legend()
        plt.savefig(os.path.join(output_path, 'acf_{}.pdf'.format(k)), format='pdf')
        plt.close()

    acf_error_all = np.array(acf_error_all)

    fig, axs = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=df, x='Time Lag', y='ACF', hue='Type', ci='sd', hue_order=['True system', 'NN system'])
    plt.legend()
    plt.savefig(os.path.join(output_path, 'acf_combined.pdf'), format='pdf')
    plt.close()
    logger.info('ACF MSE = {} +/- {} (T_long = {}) [took {} sec]'.format(np.mean(acf_error_all), np.std(acf_error_all), T_long, round(default_timer() - t0_local, 2)))

#     ## compute mean of last 10 Times of long timeseries
#     n = min(nn_max, int(10 / dt))
#     logger.info('Mean of approx final T=100 {}'.format(np.mean(sol_4d_nn[-n:,:], axis=0)))

#     n = min(true_max, int(10 / dt))
#     logger.info('Mean of true final T=100 {}'.format(np.mean(sol_3d_true[-n:,:], axis=0)))

#     logger.info('X_approx(T_end) {}'.format(sol_4d_nn[-1:,:]))
#     logger.info('X_true(T_end) {}'.format(sol_3d_true[-1:,:]))

    return sol_3d_true_kde
