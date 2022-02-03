import os
import numpy as np
import pickle
from pydoc import locate
import torch
from tqdm import tqdm
import scipy

from computation_utils import computeErrors, computeTestErrors
from plotting_utils import *
from odelibrary import my_solve_ivp
import math

#BayesianOptimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from pdb import set_trace as bp

def get_Psi_ode(dt, rhs, integrator, t0=0):
	t_span = [t0, t0+dt]
	t_eval = np.array([t0+dt])
	settings = {}
	settings['dt'] = dt
	settings['method'] = integrator
	return lambda ic0: my_solve_ivp(ic=ic0, f_rhs=lambda t, y: rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)

class Psi(object):
	def __init__(self, dt, dynamics_rhs='L63', integrator='RK45', t0=0):
		self.settings = {'dt': dt, 'method': integrator}
		self.t_span = [t0, t0+dt]
		self.t_eval = np.array([t0+dt])
		ODE = locate('odelibrary.{}'.format(dynamics_rhs))
		self.ode = ODE()

	def step_wrap(self, ic):
		foo =  my_solve_ivp(ic=ic, f_rhs=lambda t, y: self.ode.rhs(y, t), t_eval=self.t_eval, t_span=self.t_span, settings=self.settings)
		return foo

class ENKF(object):
	def __init__(self,
				Psi,
				H,
				y_obs,
				dt,
				x_true=None,
				t0=0,
				K_3dvar=None,
				v0_mean=None,
				v0_cov=None,
				output_dir='default_output_EnKF',
				N_particles=100,
				obs_noise_sd_assumed_enkf=0.1,
				obs_noise_sd_assumed_3dvar=0.1,
				obs_noise_sd_true=0,
				state_noise_sd=0,
				s_perturb_obs=True,
				rng_seed=0):

		np.random.seed(rng_seed)

		self.v0_mean = v0_mean
		self.v0_cov = v0_cov
		self.v0_cov_inv = np.linalg.inv(self.v0_cov)
		# self.v0_cov_inv = np.eye(self.v0_mean.shape[0])

		self.y_obs = y_obs # define the data
		self.x_true = x_true # define true underlying state (possibly None)

		self.N_filter = y_obs.shape[0]
		self.times_filter = np.arange(self.N_filter)

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.N_particles = N_particles
		self.H = H # linear observation operator for assimilation system
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T_filter = t0 + self.dt*self.N_filter
		self.obs_noise_sd_assumed_enkf = obs_noise_sd_assumed_enkf
		self.obs_noise_sd_assumed_3dvar = obs_noise_sd_assumed_3dvar
		self.obs_noise_sd_true = obs_noise_sd_true
		self.state_noise_sd = state_noise_sd
		self.s_perturb_obs = s_perturb_obs

		if K_3dvar is None:
			K_3dvar = self.H.T / (1+self.obs_noise_sd_assumed_3dvar)
		self.K_3dvar = K_3dvar

		self.Psi_approx = Psi

		self.dim_x_approx = v0_mean.shape[0]
		dim_y = self.H.shape[0]
		self.obs_dim = dim_y
		self.hidden_dim = self.dim_x_approx - dim_y
		self.obs_noise_mean = np.zeros(dim_y)
		self.Gamma = (obs_noise_sd_assumed_enkf**2) * np.eye(dim_y) # obs_noise_cov
		self.Gamma_true = (obs_noise_sd_true**2) * np.eye(dim_y) # obs_noise_cov
		# self.y_obs = (self.H_true @ self.x_true.T).T + np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma_true, size=self.N_filter)

		# set up DA arrays
		# means
		self.x_pred_mean = np.zeros( (self.N_filter, self.dim_x_approx))
		self.y_pred_mean = np.zeros( (self.N_filter, dim_y))
		self.x_assim_mean = np.zeros_like(self.x_pred_mean)
		self.x_adhoc = np.zeros_like(self.x_pred_mean)
		self.x_assim_3dvar = np.zeros_like(self.x_pred_mean)
		self.x_pred_3dvar = np.zeros_like(self.x_pred_mean)
		self.y_pred_3dvar = np.zeros_like(self.y_pred_mean)

		# particles
		self.x_pred_particles = np.zeros( (self.N_filter, self.N_particles, self.dim_x_approx) )
		self.y_pred_particles = np.zeros( (self.N_filter, self.N_particles, dim_y) )
		self.x_assim_particles = np.zeros( (self.N_filter, self.N_particles, self.dim_x_approx) )

		#  error-collection arrays
		self.x_assim_error_mean = np.zeros_like(self.x_pred_mean)
		self.x_pred_mean_error = np.zeros_like(self.x_pred_mean)
		self.y_pred_mean_error = np.zeros_like(self.y_pred_mean)
		self.x_assim_error_particles = np.zeros( (self.N_filter, self.N_particles, self.dim_x_approx) )
		self.y_pred_error_particles = np.zeros( (self.N_filter, self.N_particles, dim_y) )
		self.x_adhoc_error = np.zeros_like(self.x_pred_mean)
		self.y_adhoc_error = np.zeros_like(self.y_pred_mean)
		self.x_assim_3dvar_error = np.zeros_like(self.x_pred_mean)
		self.y_assim_3dvar_error = np.zeros_like(self.y_pred_mean)
		self.x_pred_3dvar_error = np.zeros_like(self.x_pred_mean)
		self.y_pred_3dvar_error = np.zeros_like(self.y_pred_mean)

		# cov
		self.x_pred_cov = np.zeros((self.N_filter, self.dim_x_approx, self.dim_x_approx))

		# set up useful DA matrices
		self.Ix = np.eye(self.dim_x_approx)
		self.K_vec = np.zeros( (self.N_filter, self.dim_x_approx, dim_y) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)

		# choose ic for DA
		# x_ic_cov = (x_ic_sd**2) * np.eye(dim_x)
		# if x_ic_mean is None:
		# 	x_ic_mean = np.zeros(dim_x)

		v0 = np.random.multivariate_normal(mean=v0_mean, cov=v0_cov, size=self.N_particles)
		self.v0 = v0
		self.x_assim_particles[0] = np.copy(v0)
		self.x_pred_particles[0] = np.copy(v0)

		self.x_pred_mean[0] = np.mean(v0, axis=0)
		self.y_pred_mean[0] = self.H @ self.x_pred_mean[0]

	def roll_forward(self, ic, N, Psi_step):
		vec = np.zeros((N, ic.shape[0]))
		vec[0] = ic
		for n in range(1,N):
			vec[n] = Psi_step(vec[n-1])
		return vec

	def set_data(self, ic):
		foo = self.roll_forward(ic=ic, N=self.N_burnin + self.N_filter, Psi_step=self.Psi_true.step_wrap)
		self.x_true = foo[self.N_burnin:]

	def predict(self, ic):
		ic = torch.from_numpy(ic.astype(np.float32)[None,:])
		return np.squeeze(self.Psi_approx(ic, self.dt)[0].cpu().data.numpy())

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def update_3dvar(self, x_pred, y_obs):
		return (self.Ix - self.K_3dvar @ self.H) @ x_pred + (self.K_3dvar @ y_obs)

	def test(self):
		ic_dict = {
					'Ad hoc': self.x_adhoc[-1],
					'EnKF': self.x_assim_mean[-1],
					'3DVAR': self.x_assim_3dvar[-1],
					'3DVAR pred': self.x_pred_3dvar[-1],
					'True': self.x_true[-1]
					}
		# roll forward the predictions based on different ics
		traj_dict = [{} for _ in range(self.obs_dim)]
		for key in ic_dict:
			if key=='True':
				Psi_step = self.Psi_true
				H = self.H_true
			else:
				Psi_step = self.Psi_approx
				H = self.H
			foo_pred = self.roll_forward(ic=ic_dict[key], N=self.N_test, Psi_step=Psi_step)
			y_pred = (H @ foo_pred.T).T

			for i in range(self.obs_dim):
				traj_dict[i][key] = y_pred[:,i]

		for i in range(self.obs_dim):
			fig_path = os.path.join(self.output_dir, 'obs_test_predictions_dim{}'.format(i))
			plot_trajectories(times=self.times_test, traj_dict=traj_dict[i], fig_path=fig_path)

	def train_filter_new(self, N_passes=10, N_opt=500, lr=0.0000001, K_zero=True):

		# build a torch module for 3DVAR and use ADAM and a scheduler
		class assimilate(torch.nn.Module):
			def __init__(self, H, Sigma_inv):
				super().__init__()
				self.Sigma_inv = torch.from_numpy(Sigma_inv).type(torch.float64)
				self.H = torch.from_numpy(H)
				self.dim_x = self.H.shape[1]
				self.Ix = torch.eye(self.dim_x)
				if K_zero:
					self.K = torch.nn.Parameter(torch.zeros_like(self.H.T, dtype=torch.float))
				else:
					self.K = torch.nn.Parameter(self.H.T)

			def forward(self, x_pred, y_obs):
				x_assim = (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)
				return x_assim

			def loss(self, x_assim, x_true):
				dif = (x_assim - x_true)**2
				L_c = 0.5 * dif.T @ self.Sigma_inv @ dif
				return L_c

		var3d = assimilate(H=self.H, Sigma_inv=self.v0_cov_inv)
		optimizer = torch.optim.Adam(var3d.parameters(), lr = lr)
		# optimizer = torch.optim.SGD(var3d.parameters(), lr = lr)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		# 	optimizer, factor = 0.1, patience=205, verbose=True, min_lr = 0.0001)

		# initialize optimization at full forcing
		# self.K_3dvar = self.H.T
		self.K_3dvar = np.zeros_like(self.H.T)

		self.times_opt = np.arange(N_opt)

		# DA @ c=0, t=0 has been initialized already
		self.grad_vec = np.zeros( ((N_opt-1)*N_passes, self.K_3dvar.shape[0], self.K_3dvar.shape[1]) )
		self.grad_vec_new = np.zeros( ((N_opt-1)*N_passes, self.K_3dvar.shape[0], self.K_3dvar.shape[1]) )
		self.K_vec = np.zeros( ((N_opt-1)*N_passes, self.K_3dvar.shape[0], self.K_3dvar.shape[1]) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)
		self.K_vec[0] = np.copy(self.K_3dvar)
		self.K_vec_runningmean[0] = np.copy(self.K_3dvar)
		self.loss = np.zeros((N_opt-1)*N_passes)

		for q in tqdm(range(N_passes)):
			ic = np.random.randint(low=0, high=self.x_true.shape[0]-N_opt)

			# set up DA arrays
			self.x_pred = np.zeros_like(self.x_true[:N_opt])
			self.y_pred = np.zeros_like(self.y_obs[:N_opt])
			self.x_assim = np.zeros_like(self.x_true[:N_opt])

			# choose ic for DA
			# x_ic = self.x_true[-1]
			# x_ic = self.x_true[0] + np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov/50)
			x_ic = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov)
			self.x_assim[0] = x_ic
			self.x_pred[0] = x_ic
			self.y_pred[0] = self.H @ x_ic

			optimizer.zero_grad()
			for c in range(1, N_opt):
				# predict
				self.t_pred += self.dt
				self.x_pred[c] = self.predict(ic=self.x_assim[c-1])
				self.y_pred[c] = self.H @ self.x_pred[c]

				# compute loss old way
				m_c_old = self.update_3dvar(x_pred=self.x_pred[c], y_obs=self.y_obs[ic+c])
				# L_c = np.sum( (m_c - self.x_true[c])**2 ) #|| ||^2
				L_c = 0.5 * ((m_c_old - self.x_true[ic+c])**2).T @ self.v0_cov_inv @ (m_c_old - self.x_true[ic+c])**2
				# self.loss[q*(N_opt-1)+c-1] = L_c

				# compute loss new way
				m_c = var3d(x_pred=torch.from_numpy(self.x_pred[c]), y_obs=torch.from_numpy(self.y_obs[ic+c]))
				loss = var3d.loss(x_assim=m_c, x_true=torch.from_numpy(self.x_true[ic+c]))
				if c > 0:
					loss.backward(retain_graph=True)
					self.grad_vec_new[q*(N_opt-1)+c-1] = var3d.K.grad.data.numpy()
					print('Epoch {}, Step {}: ||grad|| = {}'.format(q, c, np.linalg.norm(var3d.K.grad.data.numpy())))
				# scheduler.step(loss)
				self.loss[q*(N_opt-1)+c-1] = loss

				# compute gradient of loss wrt K
				pred_err = (self.y_obs[ic+c,None].T - self.y_pred[c,None].T).T
				grad_loss = self.v0_cov_inv @ ( self.K_3dvar*pred_err + self.x_pred[c,None].T - self.x_true[ic+c,None].T) @ pred_err.T

				# update K
				# print('TORCH grad:', var3d.K.grad)
				# print('matt grad:', grad_loss)
				self.grad_vec[q*(N_opt-1)+c-1] = grad_loss
				self.K_3dvar -= lr * grad_loss
				# self.K_vec[q*(N_opt-1)+c-1] = np.copy(self.K_3dvar)
				self.K_vec[q*(N_opt-1)+c-1] = var3d.K.data.numpy()
				self.K_vec_runningmean[q*(N_opt-1)+c-1] = np.mean(self.K_vec[:q*(N_opt-1)+c], axis=0)

				# assimilate
				self.t_assim += self.dt
				# self.x_assim[c] = self.update_3dvar(x_pred=self.x_pred[c], y_obs=self.y_obs[ic+c])
				foo = var3d(x_pred=torch.from_numpy(self.x_pred[c]), y_obs=torch.from_numpy(self.y_obs[ic+c]))
				self.x_assim[c] = foo.data.numpy()


			torch.nn.utils.clip_grad_norm_(var3d.parameters(), max_norm=1)
			optimizer.step()
			if q%1==0 or q==(N_passes-1):
				# plot K convergence
				fig_path = os.path.join(self.output_dir, 'K_learning_sequence')
				plot_K_learning(K_vec=self.K_vec[:(q+1)*(N_opt-1)], fig_path=fig_path)

				fig_path = os.path.join(self.output_dir, 'gradK_learning_sequence')
				plot_K_learning(K_vec=self.grad_vec[:(q+1)*(N_opt-1)], fig_path=fig_path)

				fig_path = os.path.join(self.output_dir, 'gradK_new_learning_sequence')
				plot_K_learning(K_vec=self.grad_vec_new[:(q+1)*(N_opt-1)], fig_path=fig_path)

				fig_path = os.path.join(self.output_dir, 'K_learning_runningMean')
				plot_K_learning(K_vec=self.K_vec_runningmean[:(q+1)*(N_opt-1)], fig_path=fig_path)

				# plot learning error
				fig_path = os.path.join(self.output_dir, 'loss_sequence')
				plot_loss(loss=self.loss[:(q+1)*(N_opt-1)], fig_path=fig_path)

		return


	def train_filter(self, N_passes=10, N_opt=500, lr=0.001):

		# initialize optimization at full forcing
		# self.K_3dvar = self.H.T
		self.K_3dvar = np.zeros_like(self.H.T)

		self.times_opt = np.arange(N_opt)

		# DA @ c=0, t=0 has been initialized already
		self.grad_vec = np.zeros( ((N_opt-1)*N_passes, self.K_3dvar.shape[0], self.K_3dvar.shape[1]) )
		self.K_vec = np.zeros( ((N_opt-1)*N_passes, self.K_3dvar.shape[0], self.K_3dvar.shape[1]) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)
		self.K_vec[0] = np.copy(self.K_3dvar)
		self.K_vec_runningmean[0] = np.copy(self.K_3dvar)
		self.loss = np.zeros((N_opt-1)*N_passes)

		for q in tqdm(range(N_passes)):
			ic = np.random.randint(low=0, high=self.x_true.shape[0]-N_opt)

			# set up DA arrays
			self.x_pred = np.zeros_like(self.x_true[:N_opt])
			self.y_pred = np.zeros_like(self.y_obs[:N_opt])
			self.x_assim = np.zeros_like(self.x_true[:N_opt])

			# choose ic for DA
			# x_ic = self.x_true[-1]
			# x_ic = self.x_true[0] + np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov/50)
			x_ic = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov)
			self.x_assim[0] = x_ic
			self.x_pred[0] = x_ic
			self.y_pred[0] = self.H @ x_ic

			for c in range(1, N_opt):
				# predict
				self.t_pred += self.dt
				self.x_pred[c] = self.predict(ic=self.x_assim[c-1])
				self.y_pred[c] = self.H @ self.x_pred[c]

				# compute loss
				m_c = self.update_3dvar(x_pred=self.x_pred[c], y_obs=self.y_obs[ic+c])
				# L_c = np.sum( (m_c - self.x_true[c])**2 ) #|| ||^2
				L_c = 0.5 * ((m_c - self.x_true[ic+c])**2).T @ self.v0_cov_inv @ (m_c - self.x_true[ic+c])**2
				self.loss[q*(N_opt-1)+c-1] = L_c

				# compute gradient of loss wrt K
				pred_err = (self.y_obs[ic+c,None].T - self.y_pred[c,None].T).T
				grad_loss = self.v0_cov_inv @ ( self.K_3dvar*pred_err + self.x_pred[c,None].T - self.x_true[ic+c,None].T) @ pred_err.T

				# update K
				self.grad_vec[q*(N_opt-1)+c-1] = grad_loss
				self.K_3dvar -= lr * grad_loss
				self.K_vec[q*(N_opt-1)+c-1] = np.copy(self.K_3dvar)
				self.K_vec_runningmean[q*(N_opt-1)+c-1] = np.mean(self.K_vec[:q*(N_opt-1)+c], axis=0)

				# assimilate
				self.t_assim += self.dt
				self.x_assim[c] = self.update_3dvar(x_pred=self.x_pred[c], y_obs=self.y_obs[ic+c])

			if q%10==0 or q==(N_passes-1):
				# plot K convergence
				fig_path = os.path.join(self.output_dir, 'K_learning_sequence')
				plot_K_learning(K_vec=self.K_vec[:(q+1)*(N_opt-1)], fig_path=fig_path)

				fig_path = os.path.join(self.output_dir, 'gradK_learning_sequence')
				plot_K_learning(K_vec=self.grad_vec[:(q+1)*(N_opt-1)], fig_path=fig_path)

				fig_path = os.path.join(self.output_dir, 'K_learning_runningMean')
				plot_K_learning(K_vec=self.K_vec_runningmean[:(q+1)*(N_opt-1)], fig_path=fig_path)

				# plot learning error
				fig_path = os.path.join(self.output_dir, 'loss_sequence')
				plot_loss(loss=self.loss[:(q+1)*(N_opt-1)], fig_path=fig_path)

		return

	def loop_K(self, N_warmup=100, N_test=500, N_passes=10):
		'''N_opt is length of training trajectory.
			N_search is number of learning epochs'''
		# K_list = [np.array(self.H.T), np.zeros_like(self.H.T)]
		K_list = []
		eval_list = []

		def f_blackbox(K0, K1, K2):
			# maximize this
			K = np.array([[K0, K1, K2]]).T
			foo = f(K)
			# foo = 1 - foo / (1 + foo)
			print(K, foo)
			# if math.isnan(foo):
			# 	foo = 0
			return foo

		def f_scipy(Kvec):
			# minimize this
			K = Kvec[:,None]
			foo = f(K)
			# foo = foo / (1 + foo)
			print(K, foo)
			# if math.isnan(foo):
			# 	foo = 1
			return -foo

		# ic_list = np.random.randint(low=0, high=self.x_true.shape[0]-N_warmup-N_test, size=N_passes)
		# v0_list = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov, size=N_passes)
		# def f(K, ic_list=ic_list, v0_list=v0_list, N_passes=N_passes):
		def f(K):
			K_list.append(K)
			self.K_3dvar = K
			sub_list = []
			for q in range(N_passes):
				# ic = ic_list[q]
				# v0 = v0_list[q]
				ic = np.random.randint(low=0, high=self.x_true.shape[0]-N_warmup-N_test)
				v0 = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov)
				y_obs = self.y_obs[ic:ic+N_warmup]
				x_true = self.x_true[ic:ic+N_warmup]
				goo, ic_state = self.filter_K(v0=v0, y_obs=y_obs, x_true=x_true)
				y_test = (self.H @ self.x_true[ic+N_warmup:ic+N_warmup+N_test].T).T
				y_pred = np.zeros_like(y_test)
				for j in range(N_test):
					foo_pred = self.predict(ic=ic_state)
					y_pred[j] = self.H @ foo_pred
					ic_state = foo_pred
				errfoo = computeTestErrors(target=y_test, prediction=y_pred, dt=self.dt, thresh_list=[0.1], thresh_norm=63)
				errfoo.pop('mse')
				sub_list.append(errfoo)
			df = pd.DataFrame(sub_list)
			eval_list.append(df.mean())

			return df['t_valid_0.1'].min() + df['t_valid_0.1'].mean()

		# bp()
		# for K in K_list:
		# 	out = f(K)
		# f_blackbox(1,0,0)
		pbounds = {'K0': (0.75,1),
					'K1': (-0.3,0.1),
					'K2': (-0.3,0.1)}
		optimizer = BayesianOptimization(f=f_blackbox,
										pbounds=pbounds,
										random_state=1)
		log_path = os.path.join(self.output_dir, "BayesOpt_log.json")
		logger = JSONLogger(path=log_path) #conda version doesnt have RESET feature
		optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
		optimizer.probe(params={'K0': 1, 'K1': 0, 'K2': 0}, lazy=True)
		optimizer.probe(params={'K0': 0.765, 'K1': 0.074, 'K2': 0}, lazy=True)
		optimizer.probe(params={'K0': 0.885, 'K1': -0.016, 'K2': 0.017}, lazy=True)
		optimizer.probe(params={'K0': 0.75, 'K1': 0.052, 'K2': -0.045}, lazy=True)
		optimizer.probe(params={'K0': 0.84, 'K1': -0.1, 'K2': -0.04}, lazy=True)
		optimizer.maximize(init_points=5, n_iter=300, acq='ucb')
		best_param_dict = optimizer.max['params']
		best_quality = optimizer.max['target']
		print("Optimal parameters:", best_param_dict, '(quality = {})'.format(best_quality))
		# re-setup things with optimal parameters (new realization using preferred hyperparams)
		# self.set_BO_keyval(best_param_dict)
		# plot results from validation runs
		# df = optimizer_as_df(optimizer)

		foo = pd.DataFrame(eval_list).idxmin()
		K_opt = {nm: K_list[foo[nm]] for nm in foo.index}

		# x0 = np.array([best_param_dict['K0', 'K1', 'K2'])
		# bp()
		# res = scipy.optimize.minimize(fun=f_scipy, method='Nelder-Mead', x0=np.array([0.84, -0.1, -0.04]), maxiter=30, options={'fatol':1e-3, 'xatol': 1e-3})
		# print(res.x)

		# K_opt = {nm: np.array([[0.765, 0.074, 0]]).T for nm in foo.index}
		# K_opt = {nm: np.array([[0.75, 0.052, -0.045]]).T for nm in foo.index}
		# K_opt = {nm: np.array([[1, 0, 0]]).T for nm in foo.index}
		return K_opt


	def loop_K_old(self, N_opt=1200, N_passes=3):
		'''N_opt is length of training trajectory.
			N_search is number of learning epochs'''
		# K_list = [np.array(self.H.T), np.zeros_like(self.H.T)]
		K_list = []
		eval_list = []

		def f_blackbox(K0, K1, K2):
			K = np.array([[K0, K1, K2]]).T
			foo = f(K)
			foo = 1 - foo / (1 + foo)
			print(K, foo)
			if math.isnan(foo):
				foo = 0
			return foo

		def f_scipy(Kvec):
			K = Kvec[:,None]
			foo = f(K)
			foo = foo / (1 + foo)
			print(K, foo)
			if math.isnan(foo):
				foo = 1
			return foo

		ic_list = np.random.randint(low=0, high=self.x_true.shape[0]-N_opt, size=N_passes)
		v0_list = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov, size=N_passes)
		def f(K, ic_list=ic_list, v0_list=v0_list, N_passes=N_passes):
			K_list.append(K)
			self.K_3dvar = K
			sub_list = []
			for q in range(N_passes):
				ic = ic_list[q]
				v0 = v0_list[q]
				# ic = np.random.randint(low=0, high=self.x_true.shape[0]-N_opt)
				# v0 = np.random.multivariate_normal(mean=self.v0_mean, cov=self.v0_cov)
				y_obs = self.y_obs[ic:ic+N_opt]
				x_true = self.x_true[ic:ic+N_opt]
				goo, _ = self.filter_K(v0=v0, y_obs=y_obs, x_true=x_true)
				sub_list.append(goo)
			df = pd.DataFrame(sub_list)
			eval_list.append(df.mean())

			return df['mse_pred_obs_post10'].mean()

		# bp()
		# for K in K_list:
		# 	out = f(K)
		bp()
		pbounds = {'K0': (0.75,1),
					'K1': (-0.3,0.1),
					'K2': (-0.3,0.1)}
		optimizer = BayesianOptimization(f=f_blackbox,
										pbounds=pbounds,
										random_state=1)
		log_path = os.path.join(self.output_dir, "BayesOpt_log.json")
		logger = JSONLogger(path=log_path) #conda version doesnt have RESET feature
		optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
		optimizer.probe(params={'K0': 1, 'K1': 0, 'K2': 0}, lazy=True)
		optimizer.probe(params={'K0': 0.765, 'K1': 0.074, 'K2': 0}, lazy=True)
		optimizer.probe(params={'K0': 0.885, 'K1': -0.016, 'K2': 0.017}, lazy=True)
		optimizer.probe(params={'K0': 0.75, 'K1': 0.052, 'K2': -0.045}, lazy=True)
		optimizer.probe(params={'K0': 0.84, 'K1': -0.1, 'K2': -0.04}, lazy=True)
		optimizer.maximize(init_points=5, n_iter=300, acq='ucb')
		best_param_dict = optimizer.max['params']
		best_quality = optimizer.max['target']
		print("Optimal parameters:", best_param_dict, '(quality = {})'.format(best_quality))
		# re-setup things with optimal parameters (new realization using preferred hyperparams)
		# self.set_BO_keyval(best_param_dict)
		# plot results from validation runs
		# df = optimizer_as_df(optimizer)

		foo = pd.DataFrame(eval_list).idxmin()
		K_opt = {nm: K_list[foo[nm]] for nm in foo.index}

		# x0 = np.array([best_param_dict['K0', 'K1', 'K2'])
		# bp()
		# res = scipy.optimize.minimize(fun=f_scipy, method='Nelder-Mead', x0=np.array([0.84, -0.1, -0.04]), maxiter=30, options={'fatol':1e-3, 'xatol': 1e-3})
		# print(res.x)

		# K_opt = {nm: np.array([[0.765, 0.074, 0]]).T for nm in foo.index}
		# K_opt = {nm: np.array([[0.75, 0.052, -0.045]]).T for nm in foo.index}
		# K_opt = {nm: np.array([[1, 0, 0]]).T for nm in foo.index}
		return K_opt

	def filter_K(self, v0, y_obs, x_true=None, nm=''):
		# initialize adhoc---it is already all zeros, which will be used for hidden state
		# initialize 3dvar
		N_filter = y_obs.shape[0]
		x_assim_3dvar = np.zeros((N_filter, self.dim_x_approx))
		y_pred_3dvar = np.zeros((N_filter, self.obs_dim))

		x_pred_3dvar = np.zeros_like(x_assim_3dvar)

		y_pred_3dvar_error = np.zeros(N_filter)
		x_pred_3dvar_error = np.zeros(N_filter)
		y_assim_3dvar_error = np.zeros(N_filter)
		x_assim_3dvar_error = np.zeros(N_filter)

		# set initial assimilation
		# x_assim_3dvar[0] = v0
		x_assim_3dvar[0,:self.obs_dim] = y_obs[0]

		# DA @ c=0, t=0 has been initialized already
		for c in tqdm(range(1, N_filter)):

			# 3dvar forecasts
			x_pred_3dvar[c] = self.predict(ic=x_assim_3dvar[c-1])
			y_pred_3dvar[c] = self.H @ x_pred_3dvar[c]

			# 3dvar updates
			x_assim_3dvar[c] = self.update_3dvar(x_pred=x_pred_3dvar[c], y_obs=y_obs[c])

			# compute errors on observed dimension
			y_pred_3dvar_error[c] = np.mean( (y_obs[c] - y_pred_3dvar[c])**2 )
			y_assim_3dvar_error[c] = np.mean( (y_obs[c] - self.H @ x_assim_3dvar[c])**2 )

			# compute errors on full state (if available)
			try:
				x_pred_3dvar_error[c] = np.mean( (x_true[c] - x_pred_3dvar[c])**2 )
				x_assim_3dvar_error[c] = np.mean( (x_true[c] - x_assim_3dvar[c])**2 )
			except:
				pass

		error_dict = {}
		error_dict['mse_assim_all'] = np.mean(x_assim_3dvar_error**2)
		error_dict['mse_assim_all_post10'] = np.mean(x_assim_3dvar_error[10:]**2)
		error_dict['mse_assim_obs'] = np.mean(y_assim_3dvar_error**2)
		error_dict['mse_assim_obs_post10'] = np.mean(y_assim_3dvar_error[10:]**2)

		error_dict['mse_pred_all'] = np.mean(x_pred_3dvar_error**2)
		error_dict['mse_pred_all_post10'] = np.mean(x_pred_3dvar_error[10:]**2)
		error_dict['mse_pred_obs'] = np.mean(y_pred_3dvar_error**2)
		error_dict['mse_pred_obs_post10'] = np.mean(y_pred_3dvar_error[10:]**2)

		# Observed state evaluation
		output_dir = os.path.join(self.output_dir, nm)
		os.makedirs(output_dir, exist_ok=True)

		fig_path = os.path.join(output_dir, 'assimilation_errors_obs')
		obs_eval_dict_3dvar_pred = computeErrors(target=y_obs, prediction=x_pred_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_3dvar = computeErrors(target=y_obs, prediction=x_assim_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		errors = {'3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']   }
		plot_assimilation_errors(error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

		# plot observed state synchronization (1st state only)
		fig_path = os.path.join(output_dir, 'obs_synchronization')
		traj_dict = {'3DVAR':x_assim_3dvar[:,0],
					'3DVAR pred':x_pred_3dvar[:,0],
					'True':y_obs
					}
		plot_trajectories(traj_dict=traj_dict, fig_path=fig_path)

		# Hidden/full state evaluation
		try:
			fig_path = os.path.join(output_dir, 'assimilation_errors_all')
			eval_dict_3dvar_pred = computeErrors(target=x_true, prediction=x_pred_3dvar, dt=self.dt, thresh=0.05)
			eval_dict_3dvar = computeErrors(target=x_true, prediction=x_assim_3dvar, dt=self.dt, thresh=0.05)
			errors = {'3DVAR':eval_dict_3dvar['mse'], '3DVAR pred':eval_dict_3dvar_pred['mse'] }
			plot_assimilation_errors(error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			fig_path = os.path.join(output_dir, 'assimilation_errors_hidden')
			obs_eval_dict_3dvar_pred = computeErrors(target=x_true[:,self.obs_dim:], prediction=x_pred_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_3dvar = computeErrors(target=x_true[:,self.obs_dim:], prediction=x_assim_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			errors = {'3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']  }
			plot_assimilation_errors(error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			# plot hidden state synchronization (1st state only)
			fig_path = os.path.join(output_dir, 'hidden_synchronization')
			traj_dict = {'3DVAR': x_assim_3dvar[:,self.obs_dim+1],
						'3DVAR pred': x_pred_3dvar[:,self.obs_dim+1],
						'True':x_true[:,self.obs_dim+1]
						}
			plot_trajectories(traj_dict=traj_dict, fig_path=fig_path)
		except:
			print('True and Approximate Psi dimensions do not match; cannot evaluate hidden state assimilation.')


		ic = x_assim_3dvar[-1]
		return error_dict, ic

	def filter(self):
		# initialize adhoc---it is already all zeros, which will be used for hidden state
		# initialize 3dvar to be same as ad hoc method
		self.x_assim_3dvar[0,:self.obs_dim] = self.y_obs[0]

		# DA @ c=0, t=0 has been initialized already
		for c in tqdm(range(1, self.N_filter)):

			## predict
			self.t_pred += self.dt

			## run ad-hoc method
			# run prediction/update
			ic_adhoc = np.hstack((self.y_obs[c-1], self.x_adhoc[c-1,self.obs_dim:]))
			self.x_adhoc[c] = self.predict(ic=ic_adhoc)
			# compute errors
			try:
				self.x_adhoc_error[c] = self.x_true[c] - self.x_adhoc[c]
			except:
				pass
			self.y_adhoc_error[c] = self.H @self.x_adhoc_error[c]

			## run 3dvar method
			# 3dvar forecasts
			self.x_pred_3dvar[c] = self.predict(ic=self.x_assim_3dvar[c-1])
			self.y_pred_3dvar[c] = self.H @ self.x_pred_3dvar[c]
			# 3dvar updates
			self.x_assim_3dvar[c] = self.update_3dvar(x_pred=self.x_pred_3dvar[c], y_obs=self.y_obs[c])
			# compute errors
			try:
				self.x_pred_3dvar_error[c] = self.x_true[c] - self.x_pred_3dvar[c]
				self.x_assim_3dvar_error[c] = self.x_true[c] - self.x_assim_3dvar[c]
			except:
				pass
			self.y_pred_3dvar_error[c] = self.H @ self.x_pred_3dvar_error[c]
			self.y_assim_3dvar_error[c] = self.H @ self.x_assim_3dvar_error[c]

			# if not np.array_equal(self.x_pred_3dvar[c], self.x_adhoc[c]):
			# 	pdb.set_trace()

			## run EnKF method
			# compute and store ensemble forecasts
			for n in range(self.N_particles):
				self.x_pred_particles[c,n] = self.predict(ic=self.x_assim_particles[c-1,n])
				self.y_pred_particles[c,n] = self.H @ self.x_pred_particles[c,n]
			# compute and store ensemble means
			self.x_pred_mean[c] = np.mean(self.x_pred_particles[c], axis=0)
			self.y_pred_mean[c] = self.H @ self.x_pred_mean[c]

			# track assimilation errors for post-analysis
			# EnKF
			try:
				self.x_pred_mean_error[c] = self.x_true[c] - self.x_pred_mean[c]
			except:
				pass
			self.y_pred_mean_error[c] = self.H @ self.x_pred_mean_error[c]

			# compute and store ensemble covariance
			C_hat = np.cov(self.x_pred_particles[c], rowvar=False)
			self.x_pred_cov[c] = C_hat

			## compute gains for analysis step
			S = self.H @ C_hat @ self.H.T + self.Gamma
			self.K = C_hat @ self.H.T @ np.linalg.inv(S)
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c], axis=0)

			## assimilate
			self.t_assim += self.dt
			for n in range(self.N_particles):
				# optionally perturb the observation
				y_obs_n = self.y_obs[c] + self.s_perturb_obs * np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma)

				# prediction error for the ensemble member
				self.y_pred_error_particles[c,n] = self.y_obs[c] - self.y_pred_particles[c,n]

				# update particle
				self.x_assim_particles[c,n] = self.update(x_pred=self.x_pred_particles[c,n], y_obs=y_obs_n)

				# track assimilation errors for post-analysis
				try:
					self.x_assim_error_particles[c,n] = self.x_true[c] - self.x_assim_particles[c,n]
				except:
					pass

			# compute and store ensemble means
			self.x_assim_mean[c] = np.mean(self.x_assim_particles[c], axis=0)

			# track assimilation errors for post-analysis
			try:
				self.x_assim_error_mean[c] = self.x_true[c] - self.x_assim_mean[c]
			except:
				pass

		### compute evaluation statistics

		fig_path = os.path.join(self.output_dir, 'K_sequence')
		plot_K_learning(K_vec=self.K_vec, fig_path=fig_path)

		fig_path = os.path.join(self.output_dir, 'K_mean_sequence')
		plot_K_learning(K_vec=self.K_vec_runningmean, fig_path=fig_path)

		# Observed state evaluation
		fig_path = os.path.join(self.output_dir, 'assimilation_errors_obs')
		obs_eval_dict_3dvar_pred = computeErrors(target=self.y_obs, prediction=self.x_pred_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_3dvar = computeErrors(target=self.y_obs, prediction=self.x_assim_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_enkf = computeErrors(target=self.y_obs, prediction=self.x_assim_mean[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_adhoc = computeErrors(target=self.y_obs, prediction=self.x_adhoc[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		errors = {'Ad hoc': obs_eval_dict_adhoc['mse'], 'EnKF':obs_eval_dict_enkf['mse'], '3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']   }
		plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

		# plot observed state synchronization (1st state only)
		fig_path = os.path.join(self.output_dir, 'obs_synchronization')
		traj_dict = {'Ad hoc': self.x_adhoc[:,0],
					'EnKF':self.x_assim_mean[:,0],
					'3DVAR':self.x_assim_3dvar[:,0],
					'3DVAR pred':self.x_pred_3dvar[:,0],
					'True':self.y_obs
					}
		plot_trajectories(times=self.times_filter, traj_dict=traj_dict, fig_path=fig_path)

		# Hidden/full state evaluation
		try:
			fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
			eval_dict_3dvar_pred = computeErrors(target=self.x_true, prediction=self.x_pred_3dvar, dt=self.dt, thresh=0.05)
			eval_dict_3dvar = computeErrors(target=self.x_true, prediction=self.x_assim_3dvar, dt=self.dt, thresh=0.05)
			eval_dict_enkf = computeErrors(target=self.x_true, prediction=self.x_assim_mean, dt=self.dt, thresh=0.05)
			eval_dict_adhoc = computeErrors(target=self.x_true, prediction=self.x_adhoc, dt=self.dt, thresh=0.05)
			errors = {'Ad hoc': eval_dict_adhoc['mse'], 'EnKF':eval_dict_enkf['mse'] ,'3DVAR':eval_dict_3dvar['mse'], '3DVAR pred':eval_dict_3dvar_pred['mse'] }
			plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			fig_path = os.path.join(self.output_dir, 'assimilation_errors_hidden')
			obs_eval_dict_3dvar_pred = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_pred_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_3dvar = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_assim_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_enkf = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_assim_mean[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_adhoc = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_adhoc[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			errors = {'Ad hoc': obs_eval_dict_adhoc['mse'], 'EnKF':obs_eval_dict_enkf['mse'], '3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']  }
			plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			# plot hidden state synchronization (1st state only)
			fig_path = os.path.join(self.output_dir, 'hidden_synchronization')
			traj_dict = {'Ad hoc': self.x_adhoc[:,self.obs_dim+1],
						'EnKF': self.x_assim_mean[:,self.obs_dim+1],
						'3DVAR': self.x_assim_3dvar[:,self.obs_dim+1],
						'3DVAR pred': self.x_pred_3dvar[:,self.obs_dim+1],
						'True':self.x_true[:,self.obs_dim+1]
						}
			plot_trajectories(times=self.times_filter, traj_dict=traj_dict, fig_path=fig_path)
		except:
			print('True and Approximate Psi dimensions do not match; cannot evaluate hidden state assimilation.')
