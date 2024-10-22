import numpy as np
from scipy.stats import dirichlet, gamma, uniform, poisson, beta
from scipy.special import comb
from scipy.optimize import minimize
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


def compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=5000, rho1=None, fact=100):    
    # Generating random variables for j=2,3
    ts = uniform.rvs(loc=0, scale=2000, size=(ite, 2))
    ps = uniform.rvs(loc=3000, scale=12000, size=(ite, 2))
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

def compute_expected_utility_inteligent_competitors(trace, ts, ps, c11, c21, c31, n, T, ite=5000, rho1=None, fact=100):   
    aes = gamma.rvs(.256**2/.2**2, scale=.2**2/.256, size=(ite, 2))  # so it has mean .256 and variance .04
    mean = .837
    std = .2
    cs = beta.rvs(mean*(mean*(1-mean)/std**2-1), (1-mean)*(mean*(1-mean)/std**2-1), size=(ite, 2))
    lambda23Tt = aes * (T ** cs - ts[1:] ** cs)
    lambda23_t23 = aes * ts[1:] ** cs
    e23 = poisson.rvs(lambda23_t23, size=(ite, 2))
    qs = poisson.rvs(lambda23Tt)

    # Generating buyer random variables
    w = dirichlet.rvs([1, 1, 1], size=ite)
    rho = gamma.rvs(10, scale=1/10, size=ite)

    # Generating random variables for j=1
    acs = sample_random_ac(trace, ite)
    lambda1_t1 = acs[:, 0] * ts[0] ** acs[:, 1]
    lambda1_T = acs[:, 0] * T ** acs[:, 1]
    lambda1_Tt = acs[:, 0] * (T ** acs[:, 1] - ts[0] ** acs[:, 1])
    e1 = poisson.rvs(lambda1_t1, size=ite)
    eT = poisson.rvs(lambda1_T, size=ite)
    q1 = poisson.rvs(lambda1_Tt, size=ite)

    # Computing utility
    u1 = 1 - np.exp(-rho * (-w[:, 0] * ts[0] / T - w[:, 1] * ps[0] / 5000 - w[:, 2] * q1 / eT))
    u2 = 1 - np.exp(-rho * (-w[:, 0] * ts[1] / T - w[:, 1] * ps[1] / 5000 - w[:, 2] * qs[:, 0] / eT))
    u3 = 1 - np.exp(-rho * (-w[:, 0] * ts[2] / T - w[:, 1] * ps[2] / 5000 - w[:, 2] * qs[:, 1] / eT))

    # Computing probability of choice
    pi1 = (1 / (1 + np.exp(u2-u1) + np.exp(u3-u1))).mean()
    pi2 = (1 / (1 + np.exp(u1-u2) + np.exp(u3-u2))).mean()
    pi3 = (1 / (1 + np.exp(u1-u3) + np.exp(u2-u3))).mean()
    # Estimating the cost
    c1 = fact * (c11 * ts[0] + c21 * e1 + c31 * q1).mean()
    c2 = fact * (c11 * ts[1] + c21 * e23[:, 0] + c31 * qs[:, 0]).mean()
    c3 = fact * (c11 * ts[2] + c21 * e23[:, 1] + c31 * qs[:, 1]).mean()

    # Estimating expected utility
    util1 = np.sum([comb(n, l) * pi1**l * (1 - pi1)**(n - l) * utility_func(l * ps[0] - c1, rho1) for l in range(n+1)])
    util2 = np.sum([comb(n, l) * pi2**l * (1 - pi2)**(n - l) * utility_func(l * ps[1] - c2, rho1) for l in range(n+1)])
    util3 = np.sum([comb(n, l) * pi3**l * (1 - pi3)**(n - l) * utility_func(l * ps[2] - c3, rho1) for l in range(n+1)])
    
    return np.array([util1, util2, util3])

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

    # Bayesian search for optimal p2, t2
    space = [Real(3000, 15000, name='p2'), Real(0, 2000, name='t2')] 

    # Define the objective function wrapper for skopt
    def objective_function(params, c11, c21, c31, n, T):
        p2, t2 = params
        return -compute_expected_utility_vec_random_ac(trace, t2, p2, c11, c21, c31, n, T, ite=100000)[0]

    # Define partial function for passing extra parameters to the objective function
    objective_function_partial = partial(objective_function, c11=c11, c21=c21, c31=c31, n=n, T=T)

    # Perform Bayesian optimization
    result = gp_minimize(objective_function_partial, space, n_calls=100, n_jobs=njobs)

    # Get the optimal parameters
    p2_optimal_gp, t2_optimal_gp = result.x


    # Bayesian search for optimal p3, t3
    space = [Real(3000, 15000, name='p3'), Real(0, 2000, name='t3')]

    # Define the objective function wrapper for skopt
    def objective_function(params, c11, c21, c31, n, T):
        p3, t3 = params
        return -compute_expected_utility_vec_random_ac(trace, t3, p3, c11, c21, c31, n, T, ite=100000, rho1=1e-6)[0]

    # Define partial function for passing extra parameters to the objective function
    objective_function_partial = partial(objective_function, c11=c11, c21=c21, c31=c31, n=n, T=T)

    # Perform Bayesian optimization
    result = gp_minimize(objective_function_partial, space, n_calls=100, n_jobs=njobs)

    # Get the optimal parameters
    p3_optimal_gp, t3_optimal_gp = result.x

    print("Optimal p2:", p2_optimal_gp)
    print("Optimal t2:", t2_optimal_gp)

    print("Optimal p3:", p3_optimal_gp)
    print("Optimal t3:", t3_optimal_gp)

    # Once with optimal values for p2, t2, p3, t3, grid search for optimal t1, p1

    def compute_result(t1, p1):
        util, prob, _ = compute_expected_utility_inteligent_competitors(trace, [t1, t2_optimal_gp, t3_optimal_gp], [p1, p2_optimal_gp, p3_optimal_gp], c11, c21, c31, n, T, ite=100000)
        return p1, t1, util, prob

    params = [(t1, p1) for p1 in np.linspace(3000, 15000, 100) for t1 in np.linspace(0, 2000, 100)]
    resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1) for t1, p1 in tqdm(params))
    resultados = np.array(resultados)
    np.save('resultados_inteligent_competitors.npy', resultados)


if __name__ == '__main__':
    main(njobs=66)
    