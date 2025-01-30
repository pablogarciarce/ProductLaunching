import numpy as np
from scipy.stats import dirichlet, gamma, uniform, poisson, beta
from scipy.optimize import minimize
import pymc as pm
from joblib import Parallel, delayed
from tqdm import tqdm
import arviz as az

from utils import Trace


def utility_func(x, rho1):
    if rho1 is None: 
        return x
    return (1 - np.exp(-rho1 * x)) / rho1

def knapsack_solver(utilities, prices, budgets):
    # Generar todos los subconjuntos posibles una vez
    subsets = [[0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]

    # Inicializar la cantidad total de ventas
    total_sells = 0

    # Precalcular las sumas de precios y utilidades para cada subconjunto
    prices_sums = np.array([prices[subset, :].sum(axis=0) for subset in subsets])
    utilities_sums = np.array([utilities[subset, :].sum(axis=0) for subset in subsets])

    # Iterar sobre cada presupuesto
    for budget in budgets:
        mask = prices_sums <= budget
        utilities_masked = np.where(mask, utilities_sums, -1000)
        indices = np.argmax(utilities_masked, axis=0)

        # Contar las ventas de productos cuyos índices son pares
        total_sells += (indices % 2 == 0).sum()
        
    return total_sells


def compute_expected_utility_multiple_items(trace, t1, p1, c11, c21, c31, n, T, ite=5000, rho1=None, fact=1000):
    P = 15000
    # Generating random variables for j=2,3
    ts = uniform.rvs(loc=0, scale=T, size=(ite, 2))
    ps = uniform.rvs(loc=3000, scale=P, size=(ite, 2))
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
    u2 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 0] / T - w[:, 1] * ps[:, 0] / P - w[:, 2] * qs[:, 0] / eT))
    u3 = 1 - np.exp(-rho * (-w[:, 0] * ts[:, 1] / T - w[:, 1] * ps[:, 1] / P - w[:, 2] * qs[:, 1] / eT))

    # Estimating the cost
    c1 = fact * (c11 * t1 + c21 * e1 + c31 * q1).mean()

    # Solving the knapsack problem for each buyer
    budgets = np.random.uniform(10000, 20000, (n, ite))
    total_sales = knapsack_solver(
            np.array([u1, u2, u3]), 
            np.array([p1*np.ones([ps.shape[0]]), ps[:, 0], ps[:, 1]]),
            np.array(budgets))

    # Expected utility
    s = total_sales / ite
    util = utility_func(s * p1 - c1, rho1)
    
    profit = s * p1 - c1
    
    return util, profit, s/n

def main(njobs=44):
    trace = Trace()

    c11 = 0.2
    c21 = 1
    c31 = 5
    n = 1000
    T = 2000

    # Define la función para calcular el resultado para un par (t1, p1)
    def compute_result(t1, p1):
        util, _, prob = compute_expected_utility_multiple_items(trace, t1, p1, c11, c21, c31, n, T, ite=100000)
        return p1, t1, util, prob

    # resultados generales
    params = [(t1, p1) for p1 in np.linspace(3000, 15000, 100) for t1 in np.linspace(0, 2000, 100)]
    resultados = Parallel(n_jobs=njobs)(delayed(compute_result)(t1, p1) for t1, p1 in tqdm(params))
    resultados = np.array(resultados)
    np.save('results/results_multiple_items.npy', resultados)


if __name__ == '__main__':
    main(njobs=86)
    