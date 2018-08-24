import numpy as np
import scipy.linalg
from copy import copy

class PSA_CMAES:
    def __init__(max_generations = 100, 
                 dim=10, 
                 min_pop=100, 
                 cc = -1, 
                 cs = -1, 
                 c1 = -1, 
                 cmu = -1, 
                 sigma0 = 0.5, 
                 ftol = 1e-8, 
                 xtol = 1e-8):
        self.sigma = sigma0
        self.old_sigma = copy(self.sigma)
        self.dim = dim
        self.ftol = ftol
        self.xtol = xtol
        self.mean = np.zeros(self.dim)
        self.old_mean = copy(self.mean)
        self.c_m = 1
        self.c_mu = 1
        self.c_s = 1
        self.d_s = 1
        self.c_c = 1
        self.c_l = 1
        self.alpha = 1.4
        self.beta = 0.4
        self.lambda_min = min_pop
        self.lambda_max = np.inf
        self.CovMatrix = np.eye(self.dim, dtype=np.float64)
        self.old_CovMatrix = copy(self.CovMatrix)
        self.Sigma = copy(self.sigma**2 * self.CovMatrix)
        self.lambda_ = min_pop
        self.lambda_r = min_pop
        self.weights = np.zeros(self.lambda_r)
        self.path_c = np.zeros(self.dim)
        self.path_sigma = np.zeros(self.dim)
        self.path_theta = np.zeros(self.dim)
        self.lambda_c = 0
        self.lambda_sigma = 0
        self.lambda_theta = 0
        self.mu_eff = 0
        self.gamma_sigma = 0
        self.gamma_c = 0
        self.gamma_theta = 0
        self.qsi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim ** 2))
        self.mu_eff = 1

    def evolve(population):
        self.t = 0
        while (self.f_champion_old - self.f_champion > self.ftol) and (generations < max_generations):
            mu = np.floor(self.lambda_r/2.0)
            ranges_mu = np.arange(1, mu+1)
            weights = (np.log(mu + 0.5) - np.log(ranges_mu))/(np.sum(np.log(mu + 0.5)-np.log(ranges_mu)))
            self.weights = np.hstack((weights, np.zeros(int(lambda_r - mu))))
            self.mu_eff = 1/np.sum(np.square(weights))
            self.c_s = (mu_eff + 2)/(self.dim + mu_eff + 5)
            self.d_s = 1 + 2*max(0,np.sqrt((mu_eff-1)/(self.dim + 1)) - 1) + c_s
            self.c_c = (4 + mu_eff/self.dim)/(self.dim + 4 + 2*mu_eff/self.dim)
            self.c_l = 2 / (np.square(self.dim + 1.3) + mu_eff)

            population = self.cmaes_iteration(population)
            

        return population

    def cmaes_iteration(population):
        rank_mu_update = 0
        for i, individual in enumerate(population):
            individual = self.mean + self.sigma*np.random.multivariate_normal(np.zeros(self.dim), self.CovMatrix)
            population[i] = individual
            rank_mu_update += self.weights[i] * (np.dot((individual - self.mean)[:, np.newaxis], 
                                    (individual - self.mean)[:, np.newaxis].T)) - self.CovMatrix

        self.old_mean = copy(self.mean)
        self.old_sigma = copy(self.sigma)

        self.mean = self.mean + c_mu*np.sum(self.weights[:, np.newaxis] * (population - self.mean))
        self.path_sigma = (1 - self.c_s) * self.path_sigma + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff) * \
                    np.dot(np.linalg.inv(scipy.linalg.sqrtm(self.CovMatrix)), (self.mean - self.old_mean)/self.old_sigma)

        #self.qsi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim ** 2))
        heaviside = 1 if np.linalg.norm(self.path_sigma) < (1.4 + 2/(self.dim + 1) * qsi_n * self.gamma_sigma) else 0
        self.path_c = (1 - self.c_c) * self.path_c + heaviside * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * \
                    (self.mean - self.old_mean)/self.sigma
        self.gamma_sigma = (1 - self.c_s)**2 * self.gamma_sigma + self.c_s*(2-self.c_s)
        self.gamma_c = (1 - self.c_c)**2 * self.gamma_c + heaviside * self.c_c * (2 - self.c_c)
        self.sigma = self.sigma * np.exp((self.c_s/self.d_s) * (np.linalg.norm(self.path_sigma) / self.qsi_n - np.sqrt(self.gamma_sigma)))
        rank_one_update = np.dot(self.path_sigma[:, np.newaxis], self.path_sigma[:, np.newaxis].T) - \
                    self.gamma_c * self.CovMatrix
        self.old_CovMatrix = copy(self.CovMatrix)
        self.CovMatrix = self.CovMatrix + self.c_l * rank_one_update + self.c_mu * rank_mu_update

        return population

    def vech(a):
        y = np.triu(a).flatten(order='F')

        return y[y!=0]

    def expected_value_fisher():
        value = (self.dim * self.c_m**2)/self.mu_eff + \
            ((2 * self.dim * (self.dim - self.qsi_n**2)/self.qsi_n**2) * self.gamma_sigma * (self.c_s/self.d_s) **2 + \
            0.5 * (1 + 8*self.gamma_sigma * ((self.dim - self.qsi_n**2)/self.qsi_n**2) * (self.c_s/self.d_s)**2) * \
                  ((self.dim**2 + self.dim)*self.c_mu**2/self.mu_eff +\
                   (self.dim**2 + self.dim)*self.c_c(2-self.c_c)*self.c_l*self.c_mu*self.mu_eff*np.sum(np.power(self.weights,3)) +\
                   self.c_l**2 * ( self.gamma_c**2 * self.dim**2 + (1 - 2* self.gamma_c + 2*self.gamma_c**2)* self.dim)
                   )
            )

        return value

    def get_FIM(new_m, m, new_Sigma, Sigma):
        invSigma = np.linalg(Sigma)

        for i in range(new_m.shape[0]):
            for j in range(new_m.shape[0]):

        #FIM = delta_m.T @ invSigma @ delta_m +\
        # 0.5 * np.trace(invSigma @ delta_Sigma @ invSigma @ delta_Sigma))
        return FIM