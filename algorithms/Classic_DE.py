import numpy as np

class classical_DE(object):
    
    def __init__(self, n_gens=10, n_pop=10, n_dim=5, F=0.7, Cr=0.8, bounds=[-100, 100]):
        self.n_gens=n_gens
        self.n_pop=n_pop
        self.n_dim=n_dim
        self.F=F
        self.Cr=Cr
        self.bounds=bounds
        self
        
    def get_F(self):
        return self.F
    def get_Cr(self):
        return self.Cr
    def get_bounds(self):
        return self.bounds
    def get_population(self):
        return self.population
    
    def gen_rand(self, n_size=1):
        '''
        This function return a n_size-dimensional random vector.
        '''
        return np.random.random(n_size)

    def init_population(self, pop_size, dim=1, bounds=[-100,100]):
        '''
        This function initialize the population to be use in DE
        Arguments:
        pop_size - Number of individuals (there is no default value to this yet.).
        dim - dimension of the search space (default is 1).
        bounds - The inferior and superior limits respectively (default is [-100, 100]).
        '''
        return np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, dim))

    def keep_bounds(self, pop, bounds=[-10, 10]):
        '''
        This function keep the population in the seach space
        Arguments:
        pop - Population;
        bounds - The inferior and superior limits respectively
        '''
        pop[pop<bounds[0]] = bounds[0]; pop[pop>bounds[1]] = bounds[1]
        return pop

    def rand_p(self, pop, p=1, F=0.7):
        '''
        This function is the rand/p mutation scheme, this is a generalization of rand/1 mutation scheme
         from the first DE paper (Storn and Price).
        Arguments:
        pop - Population;
        p - Number of diferences to be used;
        F - The F scale factor for the diferences (default is 0.7);
        '''

        choices = np.random.choice(pop.shape[0], 1+2*p, replace=False)
        diffs = 0
        for idiff in range(1, len(choices), 2):
            diffs += F*((pop[choices[idiff]]-pop[choices[idiff+1]]))
        return pop[choices[0]] + diffs

    def binomial_crossover(self, pop, Cr=0.5, mutation_type=rand_p, **kwargs):
        '''
        This function make the binomial crossover.
        Arguments:
        pop - Population;
        mutation_type - mutation scheme (default is ran_p);
        **kwargs - This is relative to the mutation scheme ex. rand_p needs p and F.
        '''

        K = np.random.choice(pop.shape[1])
        for ind in range(tmp.shape[0]):
            mutant = mutation_type(pop, **kwargs)
            for jnd in range(tmp.shape[1]):
                if jnd == K or gen_rand()<Cr:
                    tmp[ind][jnd] = mutant[jnd]
        return pop