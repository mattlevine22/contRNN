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
import torch
from torch.utils.data import Dataset
from odelibrary        import *
from torchdiffeq import odeint_adjoint, odeint
from dynamical_models import L63_torch
from timeit import default_timer
from pdb import set_trace as bp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeseriesDataset(Dataset):
    def __init__(self, x, times, window, target_cols, known_inits):
        self.x = torch.FloatTensor(x)
        self.known_inits = known_inits
        self.times = torch.FloatTensor(times) # times associated with data x
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        i0 = index*self.window
        i1 = (index+1)*self.window
        x = self.x[i0:i1, self.target_cols]
        times = self.times[i0:i1]
        if self.known_inits:
            y0 = self.x[i0, 1:] #complement of target_cols
            return x, times, y0
        else:
            return x, times, index

    def __len__(self):
        return int(np.floor(len(self.x) / self.window))

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())

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

    def encode_derivative(self, x):
        x /= (self.std + self.eps)
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
                        dim_x=1,
                        dim_y=2,
                        dim_output=3,
                        dim_hidden=100,
                        infer_normalizers=False,
                        activation='relu'):
                super(Paper_NN, self).__init__()

                # assign parameter dimensions
                self.dim_x = dim_x
                self.dim_y = dim_y
                self.dim_hidden = dim_hidden
                self.dim_input = self.dim_x + self.dim_y
                self.dim_output = dim_output
                self.infer_normalizers = infer_normalizers

                # create model parameters and functions
                activation_dict = {
                    'relu': torch.nn.ReLU(),
                    'gelu': torch.nn.GELU(),
                    'tanh': torch.nn.Tanh()
                }
                self.activation                = activation_dict[activation]
                self.linearCell_in             = torch.nn.Linear(self.dim_x + self.dim_y, self.dim_hidden, bias=True)
                self.linearCell_out            = torch.nn.Linear(self.dim_hidden, self.dim_output, bias=True)

                if self.infer_normalizers:
                    self.input_sd              = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=10))
                    self.input_mean            = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=10))

                    self.output_sd              = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=100))

                self.print_n_params()

            def encode(self, x):
                return (x - self.input_mean) / self.input_sd

            def decode(self, x):
                # don't need output_mean because it is already covered by linearOut's bias
                return x * self.output_sd #+ self.output_mean

            def print_n_params(self):
                foo_all = sum(p.numel() for p in self.parameters())
                foo_tble = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print('{} total parameters ({} are trainable)'.format(foo_all, foo_tble))

            def init_y0(self, N):
                y_latent = 0*np.random.uniform(low=-0.5, high=0.5, size=(N, self.dim_y))
                self.y0 = torch.nn.Parameter(torch.from_numpy(y_latent).float())

            def f_nn(self, inp):
                if self.infer_normalizers:
                    return self.decode(self.linearCell_out(self.activation(self.linearCell_in(self.encode(inp)))))
                else:
                    return self.linearCell_out(self.activation(self.linearCell_in(inp)))

            def f0(self, inp, sigma=10):
                '''takes full [observed, hidden] state vector and returns [observed,hidden] vector.
                    inputs and outputs are in original coordinates.
                    input dimensions: N x state_dims'''
                foo = torch.zeros_like(inp, dtype=torch.float)
                foo[:,0] = -sigma*inp[:,0]
                return foo

            def rhs_numpy(self, inp, t=0):
                inp = torch.FloatTensor(inp.reshape(1,-1))
                foo = self.rhs(inp).detach().numpy() # not sure why gradient is still being computed here...it is called within model.eval() in testing loop.
                return foo


            def rhs(self, inp, t=0):
                '''input dimensions: N x state_dims'''
                return self.x_normalizer.encode_derivative(self.f0(self.x_normalizer.decode(inp))) + self.f_nn(inp)
                # return self.f_nn(inp)

            def forward(self, t, x):
                '''input dimensions: N x state_dims'''
                out = self.rhs(x)
                return out


def train_model(model,
                x_train,
                X_validation,
                use_gpu=False,
                known_inits=False,
                pre_trained=False,
                do_normalization=False,
                output_dir='output',
                obs_inds=[0],
                dt_data=0.01,
                use_true_xdot=False,
                rhs_true=L63_torch,
                epochs=500,
                batch_size=3,
                window=1000,
                n_warmup=100,
                learning_rate=0.1,
                weight_decay=0, #1e-4, or 1e-5?
                step_size=100,
                gamma=0.5,
                shuffle_train_loader=False,
                shuffle_test_loader=False
            ):

    # batch_size now refers to the number of windows selected in an epoch
    # window refers to the size of a given window

    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    # set datasets as torch tensors
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(X_validation)

    if do_normalization:
        print('Doing normalization, go go!')
        x_normalizer = UnitGaussianNormalizer(x_train)
    else:
        x_normalizer = TrivialNormalizer(x_train)
    if use_gpu:
        x_normalizer.cuda()
    model.x_normalizer = x_normalizer


    # get datasizes
    ntrain = x_train.shape[0]
    ntest = X_validation.shape[0]

    train_times = dt_data*np.arange(ntrain)
    test_times = dt_data*np.arange(ntest)

    # create train/test loaders for batching
    bs_train = min(batch_size, int(ntrain/window))
    bs_test = min(batch_size, int(ntest/window))

    train_set = TimeseriesDataset(x_train, train_times, window, obs_inds, known_inits)
    test_set = TimeseriesDataset(x_test, test_times, window, obs_inds, known_inits)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs_train, shuffle=shuffle_train_loader, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs_test, shuffle=shuffle_test_loader, drop_last=True)

    # create initial conditions for each window
    my_list = ['y_latent']
    if not pre_trained and not known_inits:
        model.init_y0(N=len(train_set))
        ## build optimizer and scheduler
        latent_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))

    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    optimizer = torch.optim.Adam([{'params': base_params, 'weight_decay': weight_decay}], lr=learning_rate, weight_decay=0)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=epochs, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)

    myloss = torch.nn.MSELoss()
    # torch.autograd.set_detect_anomaly(True)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0

        batch = -1
        for x, times, idx in train_loader:
            if use_gpu:
                x , times , idx = x.cuda(), times.cuda(), idx.cuda()
            batch += 1
            optimizer.zero_grad()

            # set up data
            if known_inits:
                u0 = torch.hstack( (x[:,0,:], idx) ) # create full state vector
            else:
                u0 = torch.hstack( (x[:,0,:], model.y0[idx]) ) # create full state vector

            if use_gpu:
                u0 = u0.cuda()

            # evaluate perfect model
            if ep==0 and known_inits:
                with torch.no_grad():
                    u_pred_BEST = odeint(rhs_true, y0=u0.T, t=times[0])
                    loss_LB = myloss(x.squeeze(), u_pred_BEST[:,0,:].squeeze().T).cpu().data.numpy()
                    print('Loss of True model (OVERFITTING LB):', loss_LB)

            # run forward model
            # u_pred = odeint(model, y0=u0, t=times[0])
            u_pred = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0), t=times[0]))

            # compute losses
            loss = myloss(x.permute(1,0,2), u_pred[:, :, :model.dim_x])

            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            if ep%100==0:
                if batch <=5:
                    b=0
                    K = u_pred.data.shape[-1]
                    fig, axs = plt.subplots(nrows=K, figsize=(20, 10))
                    for k in range(K):
                        axs[k].plot(times[b].cpu().data, u_pred[:,b,k].cpu().data, label='Approx State {}'.format(k))
                        try:
                            axs[k].plot(times[b].cpu().data, x[b,:,k].cpu().data, label='True State {}'.format(k))
                        except:
                            pass
                        axs[k].legend()
                    plt.savefig(os.path.join(output_dir,'TrainPlot_Epoch{}_Batch{}'.format(ep,batch)))
                    plt.close()

        train_loss /= len(train_loader)
        if ep%1==0:
            linIn_nrm = torch.linalg.norm(model.linearCell_in.weight.cpu().data, ord=2).cpu().data.numpy()
            linOut_nrm = torch.linalg.norm(model.linearCell_out.weight.cpu().data, ord=2).cpu().data.numpy()
            print('Epoch', ep, 'Train loss:', train_loss, ', |W_in|_2 = ', linIn_nrm, ', |W_out|_2 = ', linOut_nrm, 'Time per epoch [sec] =', round(default_timer() - t1, 2))
            torch.save(model, os.path.join(output_dir, 'rnn.pt'))


        scheduler.step(train_loss)

        # validate by running off-data and predicting ahead
        model.eval()
        with torch.no_grad():
            u0 = u_pred[-1,-1,:].cpu().data

            if ep%100==0:# and ep>0:
                print('solving for IC = ', u0)
                t_eval = dt_data*np.arange(0, 5000)
                # t_span = [t_eval[0], t_eval[-1]]
                settings= {'method': 'RK45'}
                try:
                    if use_gpu:
                        sol = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0.cuda()).reshape(1,-1), t=torch.Tensor(t_eval).cuda())).cpu().data.numpy().squeeze()
                    else:
                        sol = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0).reshape(1,-1), t=torch.Tensor(t_eval))).cpu().data.numpy().squeeze()
                except:
                    print('Solver failed')
                    continue

                # sol = my_solve_ivp( u0.numpy().reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)
                fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
                axs.plot(t_eval[:len(sol)], sol)
                plt.savefig(os.path.join(output_dir,'ContinueTraining_Short_Epoch{}'.format(ep)))
                plt.close()

                T_long = 5e4
                if ep%100==0 and ep>0:
                    T_long = 5e3
                f_path = os.path.join(output_dir,'ContinueTraining_Epoch{}'.format(ep))
                # try:
                if use_gpu:
                    try:
                        test_plots(x0=u0.reshape(1,-1).cuda(), rhs_nn=model.rhs_numpy, nn_normalizer=x_normalizer, sol_3d_true=X_validation, T_long=T_long, output_path=f_path)
                    except:
                        print('Test plots failed')
                else:
                    test_plots(x0=u0.reshape(1,-1), rhs_nn=model.rhs_numpy, nn_normalizer=x_normalizer, sol_3d_true=X_validation, T_long=T_long, output_path=f_path)
            # if ep%2000==0 and ep>0:
            #     t_eval = dt_data*np.arange(0, 1e6)
            #     # t_span = [t_eval[0], t_eval[-1]]
            #     # settings= {'method': 'RK45'}
            #     try:
            #         sol = odeint(model, y0=u0.reshape(1,-1), t=torch.Tensor(t_eval)).data.numpy().squeeze()
            #     except:
            #         print('Solver failed')
            #         continue
            #     # sol = my_solve_ivp( u0.reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)
            #     fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
            #     axs.plot(t_eval[:len(sol)], sol)
            #     plt.savefig(os.path.join(output_dir,'ContinueTraining_Long_Epoch{}'.format(ep)))
            #     plt.close()
            #
            #     sns.kdeplot(sol[:,0], label='approx')
            #     sns.kdeplot(x_test[:,0], label='True')
            #     plt.legend()
            #     plt.title('L63 first coordinate KDE')
            #     plt.savefig(os.path.join(output_dir,'ContinueTraining_KDE_Epoch{}'.format(ep)))
            #     plt.close()

    return model


def test_plots(x0, rhs_nn, nn_normalizer=None, sol_3d_true=None, rhs_true=None, T_long=5e3, output_path='outputs'):

    # rhs_nn(t, y)
    # rhs_true(t,y)
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
        sol_3d_true = my_solve_ivp( x0.reshape(-1), rhs_true, t_eval, t_span, settings)

    # solve approximate 3D ODE at initial condition x0
    if nn_normalizer is None:
        if use_gpu:
            sol_4d_nn = odeint(rhs_nn, y0=x0.reshape(1,-1).cuda(), t=torch.Tensor(t_eval).cuda())
        else:
            sol_4d_nn = my_solve_ivp( x0.reshape(-1), rhs_nn, t_eval, t_span, settings)
    else:
        sol_4d_nn = my_solve_ivp( nn_normalizer.encode(x0).reshape(-1), rhs_nn, t_eval, t_span, settings)
        sol_4d_nn = nn_normalizer.decode(torch.FloatTensor(sol_4d_nn)).cpu().data.numpy()

    nn_max = len(sol_4d_nn)
    true_max = len(sol_3d_true)

    ## Plot short term trajectories
    n = min(nn_max, int(10/dt))
    fig, axs = plt.subplots(figsize=(20, 10))
    plt.plot(t_eval[:n],sol_3d_true[:n,0], label='L63 - 3D')
#     plt.plot(t_eval[:n],sol_4d_true[:n,0], label='L63 - 4D')
    plt.plot(t_eval[:n],sol_4d_nn[:n,0], label='NN - 3D')
    plt.title('L63 first coordinate (short-time)')
    plt.savefig(os.path.join(output_path, 'trajectory_short'))
    plt.legend()
    plt.close()

    ## Plot medium term trajectories
    n = min(nn_max, int(250/dt))
    fig, axs = plt.subplots(nrows=3, figsize=(20, 10))
    for k in range(3):
        axs[k].plot(t_eval[:n],sol_3d_true[:n,k], label='True 3D')
#         axs[k].plot(t_eval[:n],sol_4d_true[:n,k], label='True 4D')
        axs[k].plot(t_eval[:n],sol_4d_nn[:n,k], label='NN 3D')
        axs[k].legend()
        axs[k].set_title('x_{}'.format(k))
    plt.savefig(os.path.join(output_path, 'trajectory_medium'))
    plt.close()

    ## Plot full-length trajectories
    fig, axs = plt.subplots(nrows=3, figsize=(20, 10))
    for k in range(3):
        axs[k].plot(dt*np.arange(len(sol_3d_true)),sol_3d_true[:,k], label='True 3D')
        axs[k].plot(t_eval[:nn_max],sol_4d_nn[:nn_max,k], label='NN 3D')
        axs[k].legend()
        axs[k].set_title('x_{}'.format(k))
    plt.savefig(os.path.join(output_path, 'trajectory_long'))
    plt.close()

    ## Plot invariant measure of trajectories for full available time-window
    n = len(sol_3d_true) #int(1000/dt)
    n_burnin = int(0.1*n)
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.kdeplot(sol_3d_true[n_burnin:,0], label='L63 - 3D')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    n_burnin = int(0.1*nn_max)
    sns.kdeplot(sol_4d_nn[n_burnin:,0], label='NN - 3D')
    plt.title('L63 first coordinate KDE (all-time)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_long'))
    plt.close()

    ## Plot invariant measure of trajectories for specific time-window (e.g. pre-collapse)
    n = min(nn_max, int(100/dt))
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.kdeplot(sol_3d_true[:n,0], label='L63 - 3D')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    sns.kdeplot(sol_4d_nn[:n,0], label='NN - 3D')
    plt.title('L63 first coordinate KDE (pre-collapse, if any.)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_preCollapse'))
    plt.close()

    ## compute mean of last 10 Times of long timeseries
    n = min(nn_max, int(10 / dt))
    print('Mean of approx final T=100', np.mean(sol_4d_nn[-n:,:], axis=0))

    n = min(true_max, int(10 / dt))
    print('Mean of true final T=100', np.mean(sol_3d_true[-n:,:], axis=0))

    print('X_approx(T_end)', sol_4d_nn[-1:,:])
    print('X_true(T_end)', sol_3d_true[-1:,:])

    return
