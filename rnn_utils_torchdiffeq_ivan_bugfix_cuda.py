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
from plotting_utils import plot_logs
from dynamical_models import L63_torch
from timeit import default_timer
import logging
from pdb import set_trace as bp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeseriesDataset(Dataset):
    def __init__(self, x, times, window, n_warmup, target_cols, known_inits):
        '''x is shape T (time) x D (space) x N (trajectories)'''
        if x.ndim==2:
            # backwards compatible with 1-trajectory datasets
            x = x[:,:,None]
        self.n_traj = x.shape[2]
        self.n_times = x.shape[0]
        self.x = torch.FloatTensor(x)
        self.known_inits = known_inits
        self.times = torch.FloatTensor(times) # times associated with data x
        self.window = window
        self.n_warmup = n_warmup
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        i_traj, i0 = self.get_coord(index)
        i1 = i0 + self.window
        x = self.x[i0:i1, self.target_cols, i_traj] #.squeeze(-1)
        times = self.times[i0 + self.n_warmup : i1]
        times_all = self.times[i0:i1]
        y0_TRUE = self.x[i0, 1:, i_traj] #.squeeze(-1) #complement of target_cols
        yend_TRUE = self.x[i1-1, 1:, i_traj] #.squeeze(-1) # get endpoint condition (i.e. IC for next window)

        return x, times, times_all, index, i_traj, y0_TRUE, yend_TRUE

    def get_coord(self, index):
        N = self.len_traj()
        i_traj = int( index / N ) # select trajectory
        i0 = index - i_traj*N  # select window number within trajectory
        return i_traj, i0

    def __len__(self):
        '''total number of windows'''
        return self.x.shape[2] * self.len_traj()

    def len_traj(self):
        '''computes number of windows per trajectory'''
        n_win_ics = int(  len(self.x) - self.window + 1  )
        self.n_win_ics = n_win_ics
        return n_win_ics

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        if x.dim()==3:
            inds = (0,2)
        else:
            inds = (0)
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, inds).squeeze()
        self.std = torch.std(x, inds).squeeze()
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

    def encode_derivative(self, x):
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
                        logger=None,
                        n_layers=2,
                        use_bilinear=False,
                        use_f0=False,
                        dim_x=1,
                        dim_y=2,
                        dim_output=3,
                        dim_hidden=100,
                        infer_normalizers=False,
                        infer_ic=True,
                        activation='relu'):
                super(Paper_NN, self).__init__()

                if logger is None:
                    log_fname = os.path.join("NN_logfile.log")
                    logging.basicConfig(filename=log_fname, level=logging.INFO)
                    logger = logging.getLogger()
                self.logger = logger

                # assign parameter dimensions
                self.n_layers = n_layers
                self.use_bilinear = use_bilinear
                self.use_f0 = use_f0
                self.dim_x = dim_x
                self.dim_y = dim_y
                self.dim_hidden = dim_hidden
                self.dim_input = self.dim_x + self.dim_y
                self.dim_output = dim_output
                self.infer_normalizers = infer_normalizers
                self.infer_ic = bool(infer_ic)

                # create model parameters and functions
                activation_dict = {
                    'relu': torch.nn.ReLU(),
                    'gelu': torch.nn.GELU(),
                    'tanh': torch.nn.Tanh()
                }
                self.activation_fun = activation_dict[activation]

                self.set_model()

                if self.infer_normalizers:
                    self.input_sd              = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=10))
                    self.input_mean            = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=10))
                    self.output_sd              = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.dim_input), mean=0.0, std=100))

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
                        self.linears.append(torch.nn.Linear(self.dim_x + self.dim_y, self.dim_hidden, bias=True)) # first layer
                        if self.use_bilinear:
                            self.bilinears.append(torch.nn.Bilinear(self.dim_x + self.dim_y, self.dim_x + self.dim_y, self.dim_hidden, bias=False)) # first layer
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

            def encode(self, x):
                return (x - self.input_mean) / self.input_sd

            def decode(self, x):
                # don't need output_mean because it is already covered by linearOut's bias
                return x * self.output_sd #+ self.output_mean

            def print_n_params(self):
                foo_all = sum(p.numel() for p in self.parameters())
                foo_tble = sum(p.numel() for p in self.parameters() if p.requires_grad)
                self.logger.info('{} total parameters ({} are trainable)'.format(foo_all, foo_tble))

            def init_y0(self, size):
                '''size = (N x dim_y) where N = N_traj * (N_window_ics+1)'''
                # note that the final y0 is a free parameter (end of the last window), and only used for consistency when matching endpoints
                y_latent = 0*np.random.uniform(low=-0.5, high=0.5, size=size)
                self.y0 = torch.nn.Parameter(torch.from_numpy(y_latent).float(), requires_grad=self.infer_ic)

            def f_nn(self, inp):
                if self.infer_normalizers:
                    inp = self.encode(inp)

                for l in range(self.n_layers):
                    inp = self.activations[l](inp)
                    if self.use_bilinear:
                        inp = self.linears[l](inp) + 0.01*self.bilinears[l](inp,inp) # easy way to cope with large initialzations of bilinear cells
                    else:
                        inp = self.linears[l](inp)

                if self.infer_normalizers:
                    inp = self.decode(inp)

                return inp

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
                foo = self.f_nn(inp)
                if self.use_f0:
                    foo += self.x_normalizer.encode_derivative(self.f0(self.x_normalizer.decode(inp)))
                return  foo
                # return self.f_nn(inp)

            def forward(self, t, x):
                '''input dimensions: N x state_dims'''
                out = self.rhs(x)
                return out


def train_model(model,
                x_train,
                X_validation,
                logger=None,
                lambda_endpoints=0,
                use_gpu=False,
                known_inits=False,
                learn_inits_only=False,
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
                min_lr=0,
                patience=10,
                max_grad_norm=0,
                shuffle_train_loader=False,
                shuffle_test_loader=False,
                plot_interval=1000,
            ):

    fast_plot_interval = max(1, int(plot_interval / 10))
    # batch_size now refers to the number of windows selected in an epoch
    # window refers to the size of a given window

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
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(X_validation)

    if do_normalization:
        logger.info('Doing normalization, go go!')
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
    bs_train = get_bs(x_train, batch_size, window)
    bs_test = get_bs(x_test, batch_size, window)

    train_set = TimeseriesDataset(x_train, train_times, window, n_warmup, obs_inds, known_inits)
    test_set = TimeseriesDataset(x_test, test_times, window, n_warmup, obs_inds, known_inits)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs_train, shuffle=shuffle_train_loader, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs_test, shuffle=shuffle_test_loader, drop_last=True)

    # create initial conditions for each window
    my_list = ['y0']
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    opt_param_list = [{'params': base_params, 'weight_decay': weight_decay, 'learning_rate': learning_rate}]
    if not known_inits:
        if not pre_trained:
            model.init_y0(size=(train_set.n_traj*(train_set.n_win_ics+1), model.dim_y)) #(N_traj*N_window_ics x dim_y) +1 for the endpoint of each trajectory
        ## build optimizer and scheduler
        latent_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
        opt_param_list.append({'params': latent_params, 'weight_decay': 0, 'learning_rate': learning_rate})

    # optimizer = torch.optim.Adam([{'params': base_params, 'weight_decay': weight_decay}], lr=learning_rate, weight_decay=0)
    optimizer = torch.optim.Adam(opt_param_list, lr=learning_rate, weight_decay=weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=epochs, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, min_lr=min_lr, patience=patience)

    lr_history = {key: [] for key in range(len(optimizer.param_groups))}
    train_loss_history = []
    train_loss_mse_history = []
    grad_norm_history = []
    myloss = torch.nn.MSELoss()
    for ep in range(epochs):
        for grp in range(len(optimizer.param_groups)):
            lr_history[grp].append(optimizer.param_groups[grp]['lr'])
        model.train()
        t1 = default_timer()
        train_loss = 0
        train_loss_mse = 0
        grad_norm = 0

        batch = -1
        for x, times, times_all, idx, i_traj, y0_TRUE, yend_TRUE in train_loader:
            if known_inits:
                y0 = y0_TRUE
                yend = yend_TRUE
            else:
                y0 = model.y0[idx]
                yend = model.y0[idx+1]
            if use_gpu:
                x, times, times_all, y0, yend, y0_TRUE, yend_TRUE = x.cuda(), times.cuda(), times_all.cuda(), y0.cuda(), yend.cuda(), y0_TRUE.cuda(), yend_TRUE.cuda()

            batch += 1
            optimizer.zero_grad()

            # set up initial condition
            u0 = torch.hstack( (x[:,0,:], y0) ) # create full state vector
            if use_gpu:
                u0 = u0.cuda()

            # evaluate perfect model
            if ep==0 and (known_inits or learn_inits_only):
                u0_TRUE = torch.hstack( (x[:,0,:], y0_TRUE) ) # create full state vector
                with torch.no_grad():
                    u_pred_BEST = odeint(rhs_true, y0=u0_TRUE.T, t=times_all[0])
                    loss_LB = myloss(x.squeeze(), u_pred_BEST[:,0,:].squeeze().T).cpu().data.numpy()
                    end_point_loss = lambda_endpoints * myloss(yend, u_pred_BEST[-1, model.dim_x:, :].T).cpu().data.numpy()
                    loss_LB += end_point_loss
                    logger.info('Loss of True model (OVERFITTING LB): {}'.format(loss_LB))
            # run forward model
            # u_pred = odeint(model, y0=u0, t=times[0])

            # run warmup
            for j in range(n_warmup):
                if learn_inits_only:
                    u0 = x_normalizer.decode(odeint(rhs_true, y0=x_normalizer.encode(u0).T, t=torch.Tensor([0, dt_data]))).permute(0,2,1)[-1]
                else:
                    u0 = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0), t=torch.Tensor([0, dt_data])))[-1]
                u0 = torch.hstack( (x[:,j+1,:], u0[:,model.dim_x:]) )
            if learn_inits_only:
                u_pred = x_normalizer.decode(odeint(rhs_true, y0=x_normalizer.encode(u0).T, t=times[0]).permute(0,2,1))
            else:
                u_pred = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0), t=times[0]))

            # compute losses
            loss = myloss(x[:,n_warmup:,:].permute(1,0,2), u_pred[:, :, :model.dim_x])
            # loss = myloss(x.permute(1,0,2), u_pred[:, :, :model.dim_x])
            train_loss_mse += loss.item()
            # last point of traj should match initial condition of next window
            end_point_loss = lambda_endpoints * myloss(yend, u_pred[-1, :, model.dim_x:])
            loss += end_point_loss

            loss.backward()

            # clip gradient norm
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            train_loss += loss.item()

            # compute gradient norms for monitoring
            grad_norm += model.compute_grad_norm()

            if ep%plot_interval==0:
                if batch <=5:
                    b=0
                    K = u_pred.data.shape[-1]
                    fig, axs = plt.subplots(nrows=K, figsize=(20, 10))
                    for k in range(K):
                        axs[k].plot(times[b].cpu().data, u_pred[:,b,k].cpu().data, label='Approx State {}'.format(k))
                        try:
                            axs[k].plot(times_all[b].cpu().data, x[b,:,k].cpu().data, label='True State {}'.format(k))
                        except:
                            pass
                        axs[k].legend()
                    plt.savefig(os.path.join(plot_dir,'TrainPlot_Epoch{}_Batch{}'.format(ep,batch)))
                    plt.close()

        # regularized loss
        train_loss /= len(train_loader)
        train_loss_history += [train_loss]

        # unregularized loss
        train_loss_mse /= len(train_loader)
        train_loss_mse_history += [train_loss_mse]

        # grad norms
        grad_norm /= len(train_loader)
        grad_norm_history += [grad_norm]
        if ep%fast_plot_interval==0:
            logger.info('Epoch {}, Train loss {}, Time per epoch [sec] = {}'.format(ep, train_loss, round(default_timer() - t1, 2)))
            torch.save(model, os.path.join(output_dir, 'rnn.pt'))
        # if ep%100==0:
            plot_logs(x=train_loss_history, name=os.path.join(summary_dir,'training_history'), title='Train Loss', xlabel='Epochs')
            plot_logs(x=train_loss_mse_history, name=os.path.join(summary_dir,'training_mse_history'), title='Train Loss MSE', xlabel='Epochs')
            for grp in lr_history.keys():
                plot_logs(x=lr_history[grp], name=os.path.join(summary_dir,'learning_rates_group{}'.format(grp)), title='Learning Rate Schedule', xlabel='Epochs')
            for l in range(len(grad_norm)):
                plot_logs(x=np.array(grad_norm_history)[:,l], name=os.path.join(summary_dir,'grad_norm_layer{}'.format(l)), title='Gradient Norms', xlabel='Epochs')
        scheduler.step(train_loss)

        # validate by running off-data and predicting ahead
        model.eval()
        with torch.no_grad():
            u0 = u_pred[-1,-1,:].cpu().data

            if ep%plot_interval==0 and ep>0 and not learn_inits_only:
                logger.info('solving for IC = {}'.format(u0))
                t_eval = dt_data*np.arange(0, 5000)
                # t_span = [t_eval[0], t_eval[-1]]
                settings= {'method': 'RK45'}
                try:
                    if use_gpu:
                        sol = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0.cuda()).reshape(1,-1), t=torch.Tensor(t_eval).cuda())).cpu().data.numpy().squeeze()
                    else:
                        sol = x_normalizer.decode(odeint(model, y0=x_normalizer.encode(u0).reshape(1,-1), t=torch.Tensor(t_eval))).cpu().data.numpy().squeeze()
                except:
                    logger.info('Solver failed')
                    continue

                # sol = my_solve_ivp( u0.numpy().reshape(-1), lambda t, y: model.rhs_numpy(y,t), t_eval, t_span, settings)
                fig, axs = plt.subplots(nrows=1, figsize=(20, 10))
                axs.plot(t_eval[:len(sol)], sol)
                plt.savefig(os.path.join(plot_dir,'ContinueTraining_Short_Epoch{}'.format(ep)))
                plt.close()

                T_long = 5e4
                if ep%plot_interval==0 and ep>0:
                    T_long = 5e3
                f_path = os.path.join(plot_dir,'ContinueTraining_Epoch{}'.format(ep))
                # try:
                if use_gpu:
                    test_plots(x0=u0.reshape(1,-1).cuda(), rhs_nn=model.rhs_numpy, nn_normalizer=x_normalizer, sol_3d_true=X_validation, T_long=T_long, output_path=f_path, logger=logger)
                else:
                    test_plots(x0=u0.reshape(1,-1), rhs_nn=model.rhs_numpy, nn_normalizer=x_normalizer, sol_3d_true=X_validation, T_long=T_long, output_path=f_path, logger=logger)

    return model


def test_plots(x0, rhs_nn, nn_normalizer=None, sol_3d_true=None, rhs_true=None, T_long=5e3, output_path='outputs', logger=None):

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

    K = sol_3d_true.shape[1]
    # solve approximate 3D ODE at initial condition x0
    if nn_normalizer is None:
        if use_gpu:
            sol_4d_nn = odeint(rhs_nn, y0=x0.reshape(1,-1).cuda(), t=torch.Tensor(t_eval).cuda())
        else:
            sol_4d_nn = my_solve_ivp( x0.reshape(-1), rhs_nn, t_eval, t_span, settings)
    else:
        if use_gpu:
            sol_4d_nn = odeint(rhs_nn, y0=nn_normalizer.encode(x0).reshape(-1).cuda(), t=torch.Tensor(t_eval).cuda())
        else:
            sol_4d_nn = my_solve_ivp( nn_normalizer.encode(x0).reshape(-1), rhs_nn, t_eval, t_span, settings)
        sol_4d_nn = nn_normalizer.decode(torch.FloatTensor(sol_4d_nn)).cpu().data.numpy()

    nn_max = len(sol_4d_nn)
    true_max = len(sol_3d_true)

    ## Plot short term trajectories
    n = min(nn_max, int(10/dt))
    fig, axs = plt.subplots(figsize=(20, 10))
    plt.plot(t_eval[:n],sol_3d_true[:n,0], label='True system')
#     plt.plot(t_eval[:n],sol_4d_true[:n,0], label='L63 - 4D')
    plt.plot(t_eval[:n],sol_4d_nn[:n,0], label='NN system')
    plt.title('First coordinate (short-time)')
    plt.savefig(os.path.join(output_path, 'trajectory_short'))
    plt.legend()
    plt.close()

    ## Plot medium term trajectories
    n = min(nn_max, int(250/dt))
    fig, axs = plt.subplots(nrows=3, figsize=(20, 10))
    for k in range(K):
        axs[k].plot(t_eval[:n],sol_3d_true[:n,k], label='True 3D')
#         axs[k].plot(t_eval[:n],sol_4d_true[:n,k], label='True 4D')
        axs[k].plot(t_eval[:n],sol_4d_nn[:n,k], label='NN 3D')
        axs[k].legend()
        axs[k].set_title('x_{}'.format(k))
    plt.savefig(os.path.join(output_path, 'trajectory_medium'))
    plt.close()

    ## Plot full-length trajectories
    fig, axs = plt.subplots(nrows=3, figsize=(20, 10))
    for k in range(K):
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
    sns.kdeplot(sol_3d_true[n_burnin:,0], label='True system')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    n_burnin = int(0.1*nn_max)
    sns.kdeplot(sol_4d_nn[n_burnin:,0], label='NN system')
    plt.title('First coordinate KDE (all-time)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_long'))
    plt.close()

    ## Plot invariant measure of trajectories for specific time-window (e.g. pre-collapse)
    n = min(nn_max, int(100/dt))
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.kdeplot(sol_3d_true[:n,0], label='True system')
#     sns.kdeplot(sol_4d_true[:n,0], label='L63 - 4D')
    sns.kdeplot(sol_4d_nn[:n,0], label='NN system')
    plt.title('First coordinate KDE (pre-collapse, if any.)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'inv_preCollapse'))
    plt.close()

    ## compute mean of last 10 Times of long timeseries
    n = min(nn_max, int(10 / dt))
    logger.info('Mean of approx final T=100 {}'.format(np.mean(sol_4d_nn[-n:,:], axis=0)))

    n = min(true_max, int(10 / dt))
    logger.info('Mean of true final T=100 {}'.format(np.mean(sol_3d_true[-n:,:], axis=0)))

    logger.info('X_approx(T_end) {}'.format(sol_4d_nn[-1:,:]))
    logger.info('X_true(T_end) {}'.format(sol_3d_true[-1:,:]))

    return
