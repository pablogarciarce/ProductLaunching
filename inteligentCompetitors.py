import numpy as np
from scipy.stats import dirichlet, gamma, uniform, poisson, beta
from scipy.special import comb
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import pymc as pm
from joblib import Parallel, delayed
from tqdm import tqdm
import arviz as az

from skopt import gp_minimize
from skopt.space import Real
from functools import partial


def estimate_a_c(cum_days, errors):
    # Observation period
    T = max(cum_days)
    n = len(cum_days)

    # Define the negative log-likelihood function
    def neg_log_likelihood(params):
        a, c = params
        return a * (T**(c+1)) / (c+1) - n * np.log(a) - c * np.sum(np.log(cum_days))  

    # Initial guess for parameters
    initial_guess = [1.0, 1.0]

    # Perform the optimization
    result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')

    # Extract the estimated parameters
    a_est, c_est = result.x
    return a_est, c_est


def get_trace(a_est, c_est, cum_days, errors):
    a_prima_est = a_est / (c_est + 1)
    c_prima_est = c_est + 1
    with pm.Model() as model:    
        a_prima = pm.Gamma('a_prima', alpha=100*a_prima_est**2, beta=100*a_prima_est) 
        c_prima = pm.Gamma('c_prima', alpha=100*c_prima_est**2, beta=100*c_prima_est)
        m_t = a_prima * cum_days ** c_prima

        # a = pm.Exponential('a', lam=a_est)  
        # c = pm.Normal('c', mu=c_est, sigma=0.1)  
        # m_t = a * cum_days ** (c+1)/(c+1)

        observed = pm.Poisson('observed', mu=m_t, observed=errors)
        
        trace = pm.sample(4000, tune=2000, target_accept=0.9, return_inferencedata=True)
    return trace


def sample_random_ac(trace, num_pairs):
    n_samples = trace.posterior.sizes["draw"]  # Número de muestras
    n_chains = trace.posterior.sizes["chain"]  # Número de cadenas

    # Seleccionar aleatoriamente pares de valores
    random_indices = np.random.choice(n_samples * n_chains, num_pairs)  # Índices aleatorios

    chain_indices = random_indices // n_samples  # Índices de la cadena
    draw_indices = random_indices % n_samples  # Índices de la muestra dentro de la cadena

    # Extraer valores de la cadena posterior usando indexación avanzada
    a_values = trace.posterior["a_prima"].values[chain_indices, draw_indices]
    c_values = trace.posterior["c_prima"].values[chain_indices, draw_indices]

    # Combinar los valores a y c en pares
    random_ac = np.column_stack((a_values, c_values))
    return random_ac


def utility_func(x, rho1):
    if rho1 is None: 
        return x
    return (1 - np.exp(-rho1 * x)) / rho1

def sample_advs(ite, grid):
    samples = []
    grid_interp = RegularGridInterpolator((np.linspace(3000, 15000, 100), np.linspace(0, 2000, 100)), grid[:, 2].reshape(100, 100), method='linear')
    while len(samples) < ite:
        t = np.random.uniform(0, 2000)
        p = np.random.uniform(3000, 15000)
        u = np.random.uniform(0, grid.max())
        # Interpolate the utility value from the grid
        u_grid = grid_interp([p, t])[0]
        if u < u_grid:
            samples.append([t, p])

    ts = np.array(samples)[:, 0]
    ps = np.array(samples)[:, 1]
    return ts, ps

def compute_expected_utility_intelligent_competitors(trace, t1, p1, c11, c21, c31, n, T, ite=5000, rho1=None, fact=100):    
    grid2 = np.load('resultados.npy')
    grid3 = np.load('resultados_rho1e-6.npy')
    # Generating random variables for j=2,3
    t2, p2 = sample_advs(ite, grid2)
    t3, p3 = sample_advs(ite, grid3)
    ts = np.array([t2, t3]).transpose()
    ps = np.array([p2, p3]).transpose()
    aes = gamma.rvs(.256**2/.2**2, scale=.2**2/.256, size=(ite, 2))  # so it has mean .256 and variance .04
    mean = .837
    std = .2
    cs = beta.rvs(mean*(mean*(1-mean)/std**2-1), (1-mean)*(mean*(1-mean)/std**2-1), size=(ite, 2))
    lambda23Tt = aes * (T ** cs - ts ** cs)
    qs = poisson.rvs(lambda23Tt) 

    # Generating buyer random variables
    w = dirichlet.rvs([1, 1, 1], size=ite)
    rho = gamma.rvs(10, scale=1/10, size=ite)

    # Generating random variables for j=1
    acs = sample_random_ac(trace, ite)
    lambda1_t1 = acs[:, 0] * t1 ** acs[:, 1]
    lambda1_T = acs[:, 0] * T ** acs[:, 1]
    lambda1_Tt = acs[:, 0] * (T ** acs[:, 1] - t1 ** acs[:, 1])
    e1 = poisson.rvs(lambda1_t1, size=ite)
    eT = poisson.rvs(lambda1_T, size=ite)
    q1 = poisson.rvs(lambda1_Tt, size=ite)

    # Computing utility
    u1 = 1 - np.exp(-rho * (-w[:, 0] * t1 / T - w[:, 1] * p1 / 5000 - w[:, 2] * q1 / eT))
    u2 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 0] / T - w[:, 1] * ps[:, 0] / 5000 - w[:, 2] * qs[:, 0] / eT)) - u1
    u3 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 1] / T - w[:, 1] * ps[:, 1] / 5000 - w[:, 2] * qs[:, 1] / eT)) - u1

    # Computing probability of choice
    pi = (1 / (1 + np.exp(u2) + np.exp(u3))).mean()
    # Estimating the cost
    c1 = fact * (c11 * t1 + c21 * e1 + c31 * q1).mean()

    # Estimating expected utility
    util = np.sum([comb(n, l) * pi**l * (1 - pi)**(n - l) * utility_func(l * p1 - c1, rho1) for l in range(n+1)])
    
    profit = np.sum([comb(n, l) * pi**l * (1 - pi)**(n - l) * (l * p1 - c1) for l in range(n+1)])
    
    return util, pi, profit

def main(njobs=44):
    try:
        trace = az.from_netcdf('trace.nc')
    except FileNotFoundError:
        print('No trace found, generating new one')
        cum_days = np.array([9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250])
        errors = np.arange(len(cum_days)) + 1

        a_est, c_est = estimate_a_c(cum_days, errors)
        trace = get_trace(a_est, c_est, cum_days, errors)
        trace.to_netcdf('trace.nc')

    c11 = 0.5
    c21 = 1
    c31 = 5
    n = 1000
    T = 2000

    def compute_result(t1, p1):
        util, pi, profit = compute_expected_utility_intelligent_competitors(trace, t1, p1, c11, c21, c31, n, T, ite=100000)
        return p1, t1, util, pi, profit

    params = [(t1, p1) for p1 in np.linspace(3000, 15000, 100) for t1 in np.linspace(0, 2000, 100)]
    #resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1) for t1, p1 in tqdm(params))
    resultados = [compute_result(t1, p1) for t1, p1 in tqdm(params)]
    resultados = np.array(resultados)
    np.save('resultados_inteligent_competitors.npy', resultados)


if __name__ == '__main__':
    main(njobs=66)
    