import numpy as np
from scipy.stats import dirichlet, gamma, uniform, poisson, beta
from scipy.special import comb
from scipy.optimize import minimize
import pymc as pm
from joblib import Parallel, delayed
from tqdm import tqdm
import arviz as az

from utils import Trace


def utility_func(x, rho1):
    if rho1 is None: 
        return x
    return 1 / rho1 - np.exp(-rho1 * x - np.log(rho1))  # TODO np.exp(-rho1 * x - np.log(rho1)) por estabilidad numérica??


def compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=5000, rho1=None, fact=1000):    
    # Generating random variables for j=2,3
    t2s = T * beta.rvs(a=1, b=1, size=ite)  # 1 1 for uniform, 2 5 for early, 5 2 for late
    t3s = T * beta.rvs(a=1, b=1, size=ite) 
    ts = np.column_stack((t2s, t3s))
    P = 15000
    ps = uniform.rvs(loc=3000, scale=P, size=(ite, 2))
    #ps = 12000 * beta.rvs(a=2, b=5, size=(ite, 2)) + 3000
    aes = gamma.rvs(.256**2/.2**2, scale=.2**2/.256, size=(ite, 2))  # so it has mean .256 and variance .04
    mean = .837
    std = .2
    cs = beta.rvs(mean*(mean*(1-mean)/std**2-1), (1-mean)*(mean*(1-mean)/std**2-1), size=(ite, 2))
    lambda23Tt = aes * (T ** cs - ts ** cs)
    qs = poisson.rvs(lambda23Tt)

    # Generating buyer random variables
    w = dirichlet.rvs([1, 2, 1], size=ite)
    rho = gamma.rvs(5, scale=1, size=ite)

    # Generating random variables for j=1
    acs = trace.sample_random_ac(ite)
    lambda1_t1 = acs[:, 0] * t1 ** acs[:, 1]
    lambda1_T = acs[:, 0] * T ** acs[:, 1]
    lambda1_Tt = acs[:, 0] * (T ** acs[:, 1] - t1 ** acs[:, 1])
    e1 = poisson.rvs(lambda1_t1, size=ite)
    eT = poisson.rvs(lambda1_T, size=ite)
    q1 = poisson.rvs(lambda1_Tt, size=ite)

    # Computing utility
    u1 = 1 - np.exp(-rho * (-w[:, 0] * t1 / T - w[:, 1] * p1 / P - w[:, 2] * q1 / eT))
    u2 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 0] / T - w[:, 1] * ps[:, 0] / P - w[:, 2] * qs[:, 0] / eT)) - u1
    u3 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 1] / T - w[:, 1] * ps[:, 1] / P - w[:, 2] * qs[:, 1] / eT)) - u1

    # Computing probability of choice
    pi = (1 / (1 + np.exp(u2) + np.exp(u3))).mean()
    # Estimating the cost
    c1 = fact * (c11 * t1 + c21 * e1 + c31 * q1).mean()

    # Estimating expected utility
    util = np.sum([comb(n, l) * pi**l * (1 - pi)**(n - l) * utility_func(l * p1 - c1, rho1) for l in range(n+1)])
    
    profit = np.sum([comb(n, l) * pi**l * (1 - pi)**(n - l) * (l * p1 - c1) for l in range(n+1)])

    return util, pi, profit


def main(njobs=44):
    trace = Trace()

    c11 = 0.2
    c21 = 1
    c31 = 5
    n = 1000
    T = 2000

    # # Define la función para calcular el resultado para un par (t1, p1)
    # def compute_result(t1, p1):
    #     util, prob, _ = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=1000000)
    #     return p1, t1, util, prob
# 
    # # resultados generales
    # params = [(t1, p1) for p1 in np.linspace(3000, 15000, 100) for t1 in np.linspace(0, 2000, 100)]
    # resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1) for t1, p1 in tqdm(params))
    # resultados = np.array(resultados)
    # np.save('results/results.npy', resultados)
# 
    # # resultados resolucion
    # params = [(t1, p1) for p1 in np.linspace(6000, 10000, 100) for t1 in np.linspace(0, 2000, 100)]
    # resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1) for t1, p1 in tqdm(params))
    # resultados = np.array(resultados)
    # np.save('results/results_resol.npy', resultados)

    # # Define la función para calcular el resultado para un par (t1, p1, rho1)
    # def compute_result(t1, p1, rho1):
    #     util, prob, profit = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31, n, T, ite=1000000, rho1=rho1)
    #     return p1, t1, util, prob, profit, rho1
    # # resultados rho
    # params = [(t1, p1, rho1) 
    #           for p1 in np.linspace(6000, 11000, 50) 
    #           for t1 in np.linspace(0, 2000, 100)
    #           for rho1 in [None, 1e-8, 1e-7, 1e-6, 5e-6, 1e-5]]
    # resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1, rho1) for t1, p1, rho1 in tqdm(params))
    # resultados = np.array(resultados)
    # np.save('results/results_rho2.npy', resultados)
    # Define la función para calcular el resultado para un par (t1, p1, c31)
    def compute_result(t1, p1, c31_param):
        util, prob, _ = compute_expected_utility_vec_random_ac(trace, t1, p1, c11, c21, c31_param, n, T, ite=1000000)
        return p1, t1, util, prob, c31_param
    # resultados c31
    params = [(t1, p1, c31) 
              for p1 in np.linspace(3000, 15000, 100) 
              for t1 in np.linspace(0, 2000, 100)
              for c31 in np.linspace(1, 10, 10)]
    resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1, c31) for t1, p1, c31 in tqdm(params))
    resultados = np.array(resultados)
    np.save('results/results_c31.npy', resultados)


if __name__ == '__main__':
    main(njobs=86)
    