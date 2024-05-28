import numpy as np
from scipy.stats import dirichlet, gamma, uniform, poisson, norm
from scipy.special import comb
from scipy.optimize import minimize
import pymc as pm
from joblib import Parallel, delayed


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
        a_prima = pm.Gamma('a_prima', alpha=10*a_prima_est**2, beta=10*a_prima_est) 
        c_prima = pm.Gamma('c_prima', alpha=10*c_prima_est**2, beta=10*c_prima_est)
        m_t = a_prima * cum_days ** c_prima

        # a = pm.Exponential('a', lam=a_est)  
        # c = pm.Normal('c', mu=c_est, sigma=0.1)  
        # m_t = a * cum_days ** (c+1)/(c+1)

        observed = pm.Poisson('observed', mu=m_t, observed=errors)
        
        trace = pm.sample(4000, tune=2000, target_accept=0.9, return_inferencedata=True)
    return trace


def sample_random_ac(trace, num_pairs):
    n_samples = trace.posterior.sizes["draw"]  # Number of samples
    n_chains = trace.posterior.sizes["chain"]  # Number of chains

    # Select randomly
    random_indices = np.random.choice(n_samples * n_chains, num_pairs)  

    random_ac = []
    for idx in random_indices:
        chain_index = idx // n_samples 
        draw_index = idx % n_samples  
        
        # Extract values from chain
        a = trace.posterior["a_prima"].isel(chain=chain_index, draw=draw_index).item()
        c = trace.posterior["c_prima"].isel(chain=chain_index, draw=draw_index).item()
        
        random_ac.append((a, c)) 
    return np.array(random_ac)


def utility_func(x, rho1):
    if rho1 is None: 
        return x
    return (1 - np.exp(-rho1 * x)) / rho1


def compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=5000, rho1=None):    
    # Generating random variables for j=2,3
    ts = uniform.rvs(loc=0, scale=2000, size=(ite, 2))
    ps = uniform.rvs(loc=3000, scale=15000, size=(ite, 2))
    aes = norm.rvs(loc=0.256, scale=0.05, size=(ite, 2))
    aes[aes<0.01] = 0.01
    cs = norm.rvs(loc=0.837, scale=0.05, size=(ite, 2))
    lambda23T = aes * T ** cs
    lambda23t = aes * ts ** cs
    qs = poisson.rvs(lambda23T) - poisson.rvs(lambda23t)
    qs[qs<0.1] = 0

    # Generating buyer random variables
    w = dirichlet.rvs([1, 1, 1], size=ite)
    rho = gamma.rvs(5, scale=1/5, size=ite)

    # Generating random variables for j=1
    acs = sample_random_ac(trace, ite)
    lambda1_t1 = acs[:, 0] * t1 ** acs[:, 1]
    lambda1_T = acs[:, 0] * T ** acs[:, 1]
    e1 = poisson.rvs(lambda1_t1, size=ite)
    eT = poisson.rvs(lambda1_T, size=ite)
    q1 = eT - e1
    q1[q1<0.1] = 0

    # Computing utility
    u1 = 1 - np.exp(-rho * (-w[:, 0] * t1 / T - w[:, 1] * p1 / 5000 - w[:, 2] * q1 / eT))
    u2 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 0] / T - w[:, 1] * ps[:, 0] / 5000 - w[:, 2] * qs[:, 0] / eT)) - u1
    u3 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 1] / T - w[:, 1] * ps[:, 1] / 5000 - w[:, 2] * qs[:, 1] / eT)) - u1

    # Computing probability of choice
    pi = (1 / (1 + np.exp(u2) + np.exp(u3))).mean()
    # Estimating the cost
    c1 = (c11 * t1 + c21 * e1 + c31 * q1).mean()

    # Estimating expected utility
    util = np.sum([comb(n, l) * pi**l * (1 - pi)**(n - l) * utility_func(l * p1 - c1, rho1) for l in range(n+1)])

    return util, pi


def main():
    cum_days = np.array([9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250])
    errors = np.arange(len(cum_days)) + 1

    a_est, c_est = estimate_a_c(cum_days, errors)
    trace = get_trace(a_est, c_est, cum_days, errors)

    c11 = 0.5
    c21 = 1
    c31 = 5
    n = 1000
    T = 2000

    # Define la función para calcular el resultado para un par (t1, p1)
    def compute_result(t1, p1):
        util, prob = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=100000)
        return p1, t1, util, prob

    # # resultados generales
    # params = [(t1, p1) for p1 in np.linspace(3000, 15000, 100) for t1 in np.linspace(0, 2000, 100)]
    # resultados = Parallel(n_jobs=44)(delayed(compute_result)(t1, p1) for t1, p1 in params)
    # resultados = np.array(resultados)
    # np.save('resultados.npy', resultados)

    # resultados resolucion
    params = [(t1, p1) for p1 in np.linspace(9000, 11000, 50) for t1 in np.linspace(0, 2000, 50)]
    resultados = Parallel(n_jobs=44)(delayed(compute_result)(t1, p1) for t1, p1 in params)
    resultados = np.array(resultados)
    np.save('resultados_resolucion.npy', resultados)

    # # Define la función para calcular el resultado para un par (t1, p1, rho1)
    # def compute_result(t1, p1, rho1):
    #     util, prob = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=100000, rho1=rho1)
    #     return p1, t1, util, prob, rho1
    # # resultados rho
    # params = [(t1, p1, rho1) 
    #           for p1 in np.linspace(9000, 11000, 50) 
    #           for t1 in np.linspace(500, 700, 50)
    #           for rho1 in [-1e-6, -1e-7, -1e-8, None, 1e-8, 1e-7, 1e-6]]
    # resultados = Parallel(n_jobs=44)(delayed(compute_result)(t1, p1, rho1) for t1, p1, rho1 in params)
    # resultados = np.array(resultados)
    # np.save('resultados_rho.npy', resultados)

    # # Define la función para calcular el resultado para un par (t1, p1, c31)
    # def compute_result(t1, p1, c31_param):
    #     util, prob = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31_param, n, T, ite=100000)
    #     return p1, t1, util, prob, c31_param
    # # resultados c31
    # params = [(t1, p1, c31) 
    #           for p1 in np.linspace(9000, 11000, 50) 
    #           for t1 in np.linspace(500, 700, 50)
    #           for c31 in np.linspace(1, 20, 40)]
    # resultados = Parallel(n_jobs=44)(delayed(compute_result)(t1, p1, c31) for t1, p1, c31 in params)
    # resultados = np.array(resultados)
    # np.save('resultados_c31.npy', resultados)


if __name__ == '__main__':
    main()
    