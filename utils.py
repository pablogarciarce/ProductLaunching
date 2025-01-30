import numpy as np
import pymc as pm
import arviz as az
from scipy.optimize import minimize

# SOLVING LUSTRE PROBLEM
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

class Trace:
    def __init__(self, trace_file='trace.nc'):
        self.trace_file = trace_file

    def estimate_a_c(self, cum_days, errors):
        T = max(cum_days)
        n = len(cum_days)

        def neg_log_likelihood(params):
            a, c = params
            return a * (T**(c+1)) / (c+1) - n * np.log(a) - c * np.sum(np.log(cum_days))  

        initial_guess = [1.0, 1.0]
        result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')
        a_est, c_est = result.x
        return a_est, c_est

    def get_trace(self):
        try:
            trace = az.from_netcdf(self.trace_file, engine='h5netcdf')
        except FileNotFoundError:
            print('No trace found, generating new one')
            cum_days = np.array([9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250])
            errors = np.arange(len(cum_days)) + 1

            a_est, c_est = self.estimate_a_c(cum_days, errors)
            trace = self._compute_trace(a_est, c_est, cum_days, errors)
            trace.to_netcdf(self.trace_file)
        return trace

    def _compute_trace(self, a_est, c_est, cum_days, errors):
        a_prima_est = a_est / (c_est + 1)
        c_prima_est = c_est + 1
        
        with pm.Model() as model:    
            a_prima = pm.Gamma('a_prima', alpha=100 * a_prima_est**2, beta=100 * a_prima_est) 
            c_prima = pm.Gamma('c_prima', alpha=100 * c_prima_est**2, beta=100 * c_prima_est)
            m_t = a_prima * cum_days ** c_prima
            
            observed = pm.Poisson('observed', mu=m_t, observed=errors)
            
            trace = pm.sample(4000, tune=2000, target_accept=0.9, return_inferencedata=True)
        
        return trace

    def sample_random_ac(self, num_pairs):
        trace = self.get_trace()
        n_samples = trace.posterior.sizes["draw"]
        n_chains = trace.posterior.sizes["chain"]

        random_indices = np.random.choice(n_samples * n_chains, num_pairs)
        chain_indices = random_indices // n_samples
        draw_indices = random_indices % n_samples

        a_values = trace.posterior["a_prima"].values[chain_indices, draw_indices]
        c_values = trace.posterior["c_prima"].values[chain_indices, draw_indices]

        random_ac = np.column_stack((a_values, c_values))
        return random_ac
