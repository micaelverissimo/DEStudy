__all__ = ['NSSDE']
import numpy as np

def gen_rand(n_size=1):
    '''
    This function return a n_size-dimensional random vector.
    '''
    return np.random.random(n_size)

class NSSDE(object):
    
    def __init__(self, n_gens=10000, n_pop=100, n_dim=30, F=0.8, Cr=0.9,
                 bounds=[-100, 100], scheme='rand/p/bin', p=1, global_max= 600., fitness=None):
        self.n_gens=n_gens
        self.n_pop=n_pop
        self.n_dim=n_dim
        self.F=F
        self.Cr=Cr
        self.bounds=bounds
        self.scheme=scheme
        self.p=p
        self.population=NSSDE.init_population(self, pop_size=self.n_pop,
                                                     dim=self.n_dim, bounds=self.bounds)
        self.fitness=fitness
        self.MaxEF= 10000*self.n_dim
        self.F_evals = 0
        self.global_max = global_max
        
    def get_F(self):
        return self.F
    def get_Cr(self):
        return self.Cr
    def get_p(self):
        return self.p
    def get_fitness(self):
        return self.fitness
    def get_scheme(self):
        return self.scheme
    def get_n_gens(self):
        return self.n_gens
    def get_bounds(self):
        return self.bounds
    def get_population(self):
        return self.population

    def init_population(self, pop_size, dim, bounds):
        '''
        This function initialize the population to be use in DE
        Arguments:
        pop_size - Number of individuals (there is no default value to this yet.).
        dim - dimension of the search space (default is 1).
        bounds - The inferior and superior limits respectively (default is [-100, 100]).
        '''
        return np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, dim))

    def keep_bounds(self, pop, bounds):
        '''
        This function keep the population in the seach space
        Arguments:
        pop - Population;
        bounds - The inferior and superior limits respectively
        '''
        pop[pop<bounds[0]] = bounds[0]; pop[pop>bounds[1]] = bounds[1]
        return pop
    
    def evolution(self):
        r_info = {}
        # ============ Evaluate the initial population ============
        pop_fitness = np.zeros(self.population.shape[0])
        for ind in range(self.population.shape[0]):
            pop_fitness[ind] = self.fitness(self.population[ind])
        best_idx = np.argmin(pop_fitness)
        r_info['log'] = []
        r_info['log'].append((self.F_evals, pop_fitness[best_idx], np.mean(pop_fitness),
                              np.std(pop_fitness), np.max(pop_fitness), np.median(pop_fitness),self.F, self.Cr))
        
        while self.F_evals < self.MaxEF: 
            mutant = np.zeros_like(self.population)
            trial_pop = np.copy(self.population)
            trial_fitness = np.zeros(trial_pop.shape[0])
            
            for ind in range(self.population.shape[0]):
                
                # ============ Adapt F and Cr ============
                NF = self.F
                NCr = self.Cr
                if gen_rand() < 0.1:
                    NF = 0.2 +0.2*gen_rand()
                    NCr = 0.8 +0.2*gen_rand()
                # ============ Mutation Step ============
                tmp_pop = np.delete(self.population, ind, axis=0)
                choices = np.random.choice(tmp_pop.shape[0], 1+2*self.p, replace=False)
                diffs = 0
                for idiff in range(1, len(choices), 2):
                    diffs += NF*((tmp_pop[choices[idiff]]-tmp_pop[choices[idiff+1]]))
                mutant[ind] = tmp_pop[choices[0]] + diffs
                # keep the bounds
                mutant = NSSDE.keep_bounds(self, mutant, bounds=self.bounds)
            
                # ============ Crossover Step ============               
                K = np.random.choice(trial_pop.shape[1])
                for jnd in range(trial_pop.shape[1]):
                    if jnd == K or gen_rand()<NCr:
                        trial_pop[ind][jnd] = mutant[ind][jnd]
                # keep the bounds
                trial_pop = NSSDE.keep_bounds(self, trial_pop, bounds=self.bounds)
            
                trial_fitness[ind] = self.fitness(trial_pop[ind])
                self.F_evals += 1
                if self.F_evals > self.MaxEF-1:
                    r_info['Population'] = self.population
                    r_info['Fitness'] = pop_fitness
                    r_info['Champion'] = pop_fitness[best_idx]
                    r_info['Champion Index'] = best_idx
                    r_info['Function Evals'] = self.F_evals
                    return r_info
                    
                # ============ Selection ============
                if trial_fitness[ind] < pop_fitness[ind]:
                    self.population[ind] = trial_pop[ind]
                    pop_fitness[ind] = trial_fitness[ind]
                    self.F = NF
                    self.Cr = NCr
                    if trial_fitness[ind] < pop_fitness[best_idx]:
                        best_idx = ind
                # Save Log
                r_info['log'].append((self.F_evals, pop_fitness[best_idx], np.mean(pop_fitness),
                              np.std(pop_fitness), np.max(pop_fitness), np.median(pop_fitness),self.F, self.Cr))
            # ========== Local Search =============
            a_1 = gen_rand(); a_2 = gen_rand()
            a_3 = 1.0 - a_1 - a_2
            
            k, r1, r2 = np.random.choice(self.population.shape[0], size=3)
            V = np.zeros_like(self.population[k])
            for jdim in range(self.population.shape[1]):
                V[jdim] = a_1*self.population[k][jdim] + a_2*self.population[best_idx][jdim] + a_3*(self.population[r1][jdim] - self.population[r2][jdim])
                V = NSSDE.keep_bounds(self, V, bounds=self.bounds)

            self.F_evals += 1
            F_V = self.fitness(V)
            if F_V < pop_fitness[k]:
                self.population[k] = V
                pop_fitness[k] = F_V
                if F_V < pop_fitness[best_idx]:
                    best_idx = k
            # Save Log
            r_info['log'].append((self.F_evals, pop_fitness[best_idx], np.mean(pop_fitness),
                                 np.std(pop_fitness), np.max(pop_fitness), np.median(pop_fitness),self.F, self.Cr))
            # Check the stop criteria
            if np.abs(pop_fitness[best_idx] - self.global_max)<1e-6:
                print('Stop criteria... ')
                r_info['Population'] = self.population
                r_info['Fitness'] = pop_fitness
                r_info['Champion'] = pop_fitness[best_idx]
                r_info['Champion Index'] = best_idx
                r_info['Function Evals'] = self.F_evals
                return r_info

        r_info['Population'] = self.population
        r_info['Fitness'] = pop_fitness
        r_info['Champion'] = pop_fitness[best_idx]
        r_info['Champion Index'] = best_idx
        r_info['Function Evals'] = self.F_evals
        return r_info