import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold

def gen_rand(n_size=1):
    '''
    This function return a n_size-dimensional random vector.
    '''
    return np.random.random(n_size)

class NN_DE(object):
    
    def __init__(self, n_pop=10, n_neurons=5, F=0.2, Cr=0.9, p=1, change_scheme=True ,scheme='rand',
                 bounds=[-1, 1], max_sp_evals=np.int(1e5)):
        #self.n_gens=n_gens
        self.n_pop=n_pop
        self.n_neurons=n_neurons
        self.F=F*np.ones(self.n_pop)
        self.Cr=Cr*np.ones(self.n_pop)
        self.bounds=bounds
        self.p=p
        self.scheme=scheme
        self.change_schame=change_scheme
        self.max_sp_evals=max_sp_evals
        self.sp_evals=0
        self.interactions=0
        # Build generic model
        model = Sequential()
        model.add(Dense(self.n_neurons, input_dim=100, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile( loss='mean_squared_error', optimizer = 'rmsprop', metrics = ['accuracy'] )
        self.model=model
        self.change_schame=False
        self.n_dim=model.count_params()
        #self.population=NN_DE.init_population(self, pop_size=self.n_pop,
        #                                             dim=self.n_dim, bounds=self.bounds)
        #self.train_dataset= train_dataset
        #self.test_dataset= test_dataset
        
    def init_population(self, pop_size, dim, bounds=[-1,1]):
        '''
        This function initialize the population to be use in DE
        Arguments:
        pop_size - Number of individuals (there is no default value to this yet.).
        dim - dimension of the search space (default is 1).
        bounds - The inferior and superior limits respectively (default is [-1, 1]).
            '''
        return np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, dim))

    def keep_bounds(self, pop, bounds, idx):
        '''
        This function keep the population in the seach space
        Arguments:
        pop - Population;
        bounds - The inferior and superior limits respectively
        '''
        #up_ = np.where(pop>bounds[1])
        #down_ = np.where(pop<bounds[1])
        #best_ = pop[idx]
        #print(pop[pop<bounds[0]])
        #print(down_)
        #print(best_.shape)
        pop[pop<bounds[0]] = bounds[0]; pop[pop>bounds[1]] = bounds[1]
        #pop[pop<bounds[0]] = 0.5*(bounds[0]+best_[down_]); pop[pop>bounds[1]] = 0.5*(bounds[1]+best_[up_])
        return pop

    # Define the Fitness to be used in DE
    def sp_fitness(self, target, score):
        '''
        Calculate the SP index and return the index of the best SP found

        Arguments:
        target: True labels
        score: the predicted labels
        '''
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(target, score)
        jpr = 1. - fpr
        sp = np.sqrt( (tpr  + jpr)*.5 * np.sqrt(jpr*tpr) )
        idx = np.argmax(sp)
        return sp[idx], tpr[idx], fpr[idx]#sp, idx, sp[idx], tpr[idx], fpr[idx]

    def convert_vector_weights(self, pop, nn_model):
        
        model = nn_model
        
        generic_weights = model.get_weights()
        hl_lim = generic_weights[0].shape[0]*generic_weights[0].shape[1]
        
        w = []
        hl = pop[:hl_lim]
        ol = pop[hl_lim+generic_weights[1].shape[0]:hl_lim+generic_weights[1].shape[0]+generic_weights[1].shape[0]] 
        w.append(hl.reshape(generic_weights[0].shape))
        w.append(pop[hl_lim:hl_lim+generic_weights[1].shape[0]])
        w.append(ol.reshape(generic_weights[2].shape))
        w.append(np.array(pop[-1]).reshape(generic_weights[-1].shape))
        
        return w
        
    def set_weights_to_keras_model_and_compute_fitness(self,pop, data, test_data, nn_model):
        '''
        This function will create a generic model and set the weights to this model and compute the fitness.
        Arguments:
        pop - The population of weights.
        data - The samples to be used to test.
        '''
        fitness = np.zeros((pop.shape[0],3))
        test_fitness = np.zeros((pop.shape[0],3))
        model=nn_model
        for ind in range(pop.shape[0]):
            w = NN_DE.convert_vector_weights(self, pop=pop[ind], nn_model=model)
            model.set_weights(w)
            y_score = model.predict(data[0])
            fitness[ind] = NN_DE.sp_fitness(self, target=data[1], score=y_score)
            
            # Compute the SP for test in the same calling to minimeze the evals
            test_y_score = model.predict(test_data[0])
            test_fitness[ind] = NN_DE.sp_fitness(self, target=test_data[1], score=test_y_score)
            #print('Population ind: {} - SP: {} - PD: {} - PF: {}'.format(ind, fitness[ind][0], fitness[ind][1], fitness[ind][2]))
        return fitness, test_fitness


    def evolution(self, train_dataset, test_dataset):
        
        self.population=NN_DE.init_population(self, pop_size=self.n_pop,
                                              dim=self.n_dim, bounds=self.bounds)
        r_NNDE = {}
        fitness, test_fitness = NN_DE.set_weights_to_keras_model_and_compute_fitness(self, pop=self.population,
                                                                       data=train_dataset,
                                                                       test_data=test_dataset,
                                                                       nn_model=self.model)
        best_idx = np.argmax(fitness[:,0])
        #print('Best NN found - SP: {} / PD: {} / FA: {}'.format(fitness[best_idx][0],
        #                                                        fitness[best_idx][1],
        #                                                        fitness[best_idx][2]))
        #print('Test > Mean - SP: {} +- {}'.format(np.mean(test_fitness,axis=0)[0],
        #                                                            np.std(test_fitness,axis=0)[0]))
        
        # Create the vectors F and Cr to be adapted during the interactions
        NF = np.zeros_like(self.F)
        NCr = np.zeros_like(self.Cr)
        # Create a log
        r_NNDE['log'] = []
        r_NNDE['log'].append((self.sp_evals, fitness[best_idx], np.mean(fitness, axis=0),
                             np.std(fitness, axis=0), np.min(fitness, axis=0), np.median(fitness, axis=0), self.F, self.Cr))

        r_NNDE['test_log'] = []
        r_NNDE['test_log'].append((self.sp_evals, test_fitness[best_idx], np.mean(test_fitness, axis=0),
                             np.std(test_fitness, axis=0), np.min(test_fitness, axis=0), np.median(test_fitness, axis=0), self.F, self.Cr))

        while self.sp_evals < self.max_sp_evals:
            #print('===== Interaction: {} ====='.format(self.interactions+1))
            # ============ Mutation Step ===============
            mutant = np.zeros_like(self.population)
            for ind in range(self.population.shape[0]):
                if gen_rand() < 0.1:
                    NF[ind] = 0.2 +0.2*gen_rand()
                else:
                    NF[ind] = self.F[ind]
                tmp_pop = np.delete(self.population, ind, axis=0)
                choices = np.random.choice(tmp_pop.shape[0], 1+2*self.p, replace=False)
                diffs = 0
                for idiff in range(1, len(choices), 2):
                    diffs += NF[ind]*((tmp_pop[choices[idiff]]-tmp_pop[choices[idiff+1]]))
                    if self.scheme=='rand':
                        mutant[ind] = tmp_pop[choices[0]] + diffs
                    elif self.scheme=='best':
                        mutant[ind] = self.population[best_idx] + diffs
            # keep the bounds
            mutant = NN_DE.keep_bounds(self, mutant, bounds=[-1,1], idx=best_idx)

            # ============ Crossover Step ============= 
            trial_pop = np.copy(self.population)
            K = np.random.choice(trial_pop.shape[1])
            for ind in range(trial_pop.shape[0]):
                if gen_rand() < 0.1:
                    NCr[ind] = 0.8 +0.2*gen_rand()
                else:
                    NCr[ind] = self.Cr[ind]
                for jnd in range(trial_pop.shape[1]):
                    if jnd == K or gen_rand()<NCr[ind]:
                        trial_pop[ind][jnd] = mutant[ind][jnd]
            # keep the bounds
            trial_pop = NN_DE.keep_bounds(self, trial_pop, bounds=[-1,1], idx=best_idx)

            trial_fitness, test_fitness = NN_DE.set_weights_to_keras_model_and_compute_fitness(self, pop=trial_pop,
                                                                                               data=train_dataset,
                                                                                               test_data=test_dataset,
                                                                                               nn_model=self.model)
            self.sp_evals += self.population.shape[0]
            
            # ============ Selection Step ==============
            winners = np.where(trial_fitness[:,0]>fitness[:,0])
        
            # Auto-adtaptation of F and Cr like NSSDE
            self.F[winners] = NF[winners]
            self.Cr[winners] = NCr[winners]
            
            # Greedy Selection
            fitness[winners] = trial_fitness[winners]
            self.population[winners] = trial_pop[winners]
            best_idx = np.argmax(fitness[:,0])
            
            if self.interactions > 0.95*self.max_sp_evals/self.n_pop:
              print('=====Interaction: {}====='.format(self.interactions+1))
              print('Best NN found - SP: {} / PD: {} / FA: {}'.format(fitness[best_idx][0],
                                                                        fitness[best_idx][1],
                                                                        fitness[best_idx][2]))
              print('Test > Mean - SP: {} +- {}'.format(np.mean(test_fitness,axis=0)[0],
                                                                    np.std(test_fitness,axis=0)[0]))
            self.interactions += 1.0
            
            #if fitness[best_idx][0]>0.90 and self.change_schame==False:
            #    if self.scheme == 'best':
            #        if self.scheme!='rand':                    
            #            print('Changing the scheme to rand/p/bin')
            #        self.scheme = 'rand'
            #        self.change_schame=True
            #    else:
            #        if self.scheme!='best':                    
            #            print('Changing the scheme to best/p/bin')
            #        self.scheme = 'best'
            #        self.change_schame=True
                
            
            r_NNDE['log'].append((self.sp_evals, fitness[best_idx], np.mean(fitness, axis=0),
                             np.std(fitness, axis=0), np.min(fitness, axis=0), np.median(fitness, axis=0), self.F, self.Cr))
            
            r_NNDE['test_log'].append((self.sp_evals, test_fitness[best_idx], np.mean(test_fitness, axis=0),
                             np.std(test_fitness, axis=0), np.min(test_fitness, axis=0), np.median(test_fitness, axis=0), self.F, self.Cr))

        # Compute the test
        #test_fitness = NN_DE.set_weights_to_keras_model_and_compute_fitness(self, pop=self.population,
        #                                                               data=self.test_dataset, nn_model=self.model)

        r_NNDE['champion weights'] = NN_DE.convert_vector_weights(self, self.population[best_idx], self.model)
        r_NNDE['model'] = self.model
        r_NNDE['best index'] = best_idx
        r_NNDE['Best NN'] = fitness[best_idx]
        r_NNDE['train_fitness'] = fitness
        r_NNDE['test_fitness'] = test_fitness
        r_NNDE['population'] = self.population,
        return r_NNDE


if __name__ == '__main__':
    
    data = np.load('data17-18_13TeV.sgn_lhmedium_probes.EGAM2.bkg.vetolhvloose.EGAM7.samples.npz')
    sgn = data['signalPatterns_etBin_2_etaBin_0']
    bkg = data['backgroundPatterns_etBin_2_etaBin_0']

    sgn_trgt = np.ones(sgn.shape[0])
    bkg_trgt = -1*np.ones(bkg.shape[0])

    sgn_normalized = np.zeros_like(sgn)
    for ind in range(sgn.shape[0]):
        sgn_normalized[ind] = sgn[ind]/np.abs(np.sum(sgn[ind]))
        
    bkg_normalized = np.zeros_like(bkg)
    for ind in range(bkg.shape[0]):
        bkg_normalized[ind] = bkg[ind]/np.abs(np.sum(bkg[ind]))

    data_ = np.append(sgn_normalized, bkg_normalized, axis=0)
    trgt = np.append(sgn_trgt, bkg_trgt)

    skf = StratifiedKFold(n_splits=10)
    CVO = list(skf.split(data_, trgt))

    import multiprocessing
    import time
    nn_de = NN_DE(n_pop=20, max_sp_evals=20, scheme='rand')
    def worker(proc_id, result_dict):
        print('Work Fold: '+ str(proc_id+1))
        train_index, test_index = CVO[proc_id]
        return_dict['Fold {}'.format(proc_id+1)] = nn_de.evolution(train_dataset=(data_[train_index], trgt[train_index]),
                                                               test_dataset=(data_[test_index], trgt[test_index]))
    inicio = time.time()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for ifold in range(len(CVO)):
        p = multiprocessing.Process(target=worker, args=(ifold,return_dict))
        jobs.append(p)
        p.start()

    time.sleep(5)
    for proc in jobs:
        proc.join()

    fim=time.time()
    print('Demorou - {} segundos'.format(fim-inicio))

    #import pickle
    #with open('results_NNDE.2000SPevals.pickle', 'wb') as handle:
    #    pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
