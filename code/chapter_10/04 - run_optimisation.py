"""Complexity Economics and Sustainable Development
   Chapter 10 source code
   
   Description: runs algorithm to optimise fedral transfers

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')
import ppi




 
# Dataset
df = pd.read_csv(home+"/data/chapter_10/indicators.csv")
colYears = [col for col in df.columns if col.isnumeric()]
num_years = len(colYears)

dfb = pd.read_csv(home+"/data/chapter_10/budgets.csv")
dfb = dfb[dfb.year.isin([int(c) for c in colYears])]
dfb = dfb.groupby('stateAbbreviation').mean()
dfp = pd.read_csv(home+"/data/chapter_10/populations.csv")
pop_dict = dict([(code, dfp[dfp.stateCode==code][colYears[0:-1]].values.mean()) for code in dfb.index.values])
pop_means = [dfp[dfp.stateCode==code][colYears[0:-1]].values.mean() for code in dfb.index.values]
dfb['contributions_pc'] = dfb['contributions'].values/pop_means
dfb['total_pc'] = dfb['total'].values/pop_means


states = df.stateAbbreviation.unique()
sample_size = 100
parallel_processes = 4
sub_periods = 4
T = num_years*sub_periods



parameters = dict([(state, {}) for state in states])

for state in states:   
    
    dft = df[df.stateAbbreviation == state]
    stateCode = dft.stateCode.values[0]
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    parameters[state]['N'] = N
    
    R = (dft.instrumental.values == 'I').astype(int)
    parameters[state]['R'] = R
    
    n = R.sum()
    parameters[state]['n'] = n
    
    I0 = series[:,0]
    parameters[state]['I0'] = I0
    
    Imax = dft.Imax.values
    parameters[state]['Imax'] = Imax
    
    success_rates = ppi.get_success_rates(series)
    parameters[state]['success_rates'] = success_rates
    
    # Budget 
    total = dfb[dfb.index==state].total.values[0]
    parameters[state]['total'] = total
    
    contributions_pc = dfb[dfb.index==state].contributions_pc.values[0]
    parameters[state]['contributions_pc'] = contributions_pc
    
    contributions = dfb[dfb.index==state].contributions.values[0]
    parameters[state]['contributions'] = contributions
        
    parameters[state]['success_rates'] = success_rates
    
    B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    parameters[state]['B_dict'] = B_dict
    
    # Network
    A = np.loadtxt(home+"/data/chapter_10/networks/"+state+".csv", delimiter=',')
    parameters[state]['A'] = A
    
    # Governance
    qm = dft.monitoring.values[R==1]
    parameters[state]['qm'] = qm
    
    rl = dft.ruleOfLaw.values[R==1]
    parameters[state]['rl'] = rl
    
    parameters[state]['sdgs'] = dft.sdg.values
    
    # Parameters
    dfp = pd.read_csv(home+"/data/chapter_10/parameters/"+state+".csv")
    alphas = dfp.alpha.values
    parameters[state]['alphas'] = alphas
    
    alphas_prime = dfp.alpha_prime.values
    parameters[state]['alphas_prime'] = alphas_prime
    
    betas = dfp.beta.values
    parameters[state]['betas'] = betas






def run_state(state, Bs, sdg):
    
    I0 = parameters[state]['I0']
    alphas = parameters[state]['alphas']
    alphas_prime = parameters[state]['alphas_prime']
    betas = parameters[state]['betas']
    A = parameters[state]['A']
    R = parameters[state]['R']
    qm = parameters[state]['qm']
    rl = parameters[state]['rl']
    Imax = parameters[state]['Imax']
    B_dict = parameters[state]['B_dict']
    sdgs = parameters[state]['sdgs']
    
    sols = np.array([ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
             Bs=Bs, B_dict=B_dict, Imax=Imax) for itera in range(sample_size)])
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    
    if sdg != 'all':
        Is = np.mean(tsI, axis=0)[sdgs==sdg,-1]
    else:
        Is = np.mean(tsI, axis=0)[:,-1]

    return Is
    




def fobj(shares, type_transfer, sdg):
    
    transfers = sum([parameters[state][type_transfer] for state in states])
    dist_transfers = (shares/shares.sum())*transfers
    
    fitnesses = []
    for i, state in enumerate(states):
        
        B0s = parameters[state]['total'] - parameters[state][type_transfer] + dist_transfers[i]
        B0s = np.array([np.ones(num_years)*B0s/pop_dict[state]])
        Bs = ppi.get_dirsbursement_schedule(B0s, B_dict, T)
        
        fitness = run_state(state, Bs, sdg).mean()
        fitnesses.append(fitness)
            
    return np.mean(fitnesses)





    

popsize = 24
njobs = 2
mut=0.08
crossp=0.7

best_sol = None

all_sdgs = ['all'] + df.sdg.unique().tolist()

print('Finding via DE...')


type_transfer = 'contributions'

    
def optimize(sdg):
    
    bounds = np.array(list(zip(.0001*np.ones(32), .99*np.ones(32))))
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    dimensions = len(bounds)
    pop =  np.random.rand(popsize, dimensions)*.8 + .2
    pop[0] = np.array([parameters[state][type_transfer] for state in states])/sum([parameters[state][type_transfer] for state in states])
    best_fitness = -10
    outputs = []
    
    for step in range(100):
        
        print(sdg, step)
        
        fitness = [fobj(solution, type_transfer, sdg) for solution in pop]
        best_idx = np.argmax(fitness)
        
        if fitness[best_idx] > best_fitness:
            best_sol = pop[best_idx]
            best_fitness = fitness[best_idx]
            print(best_fitness, best_idx)
            outputs.append( [step]+(best_sol/best_sol.sum()).tolist()+[best_fitness] )
            df_sol = pd.DataFrame(outputs, columns=['step']+states.tolist()+['fitness'])
            df_sol.to_csv(home+"/data/chapter_10/simulation/optimisation/"+str(sdg)+".csv", index=False)       
    
        sorter = np.argsort(fitness)[::-1]
        survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
        new_pop = survivors.copy()
        
        for j in range(len(survivors)):
            idxs = [idx for idx in range(len(survivors)) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            new_pop.append(trial_denorm)
            
        pop = np.array(new_pop)
    

Parallel(n_jobs=parallel_processes, verbose=0)(delayed(optimize)(sdg) for sdg in all_sdgs)





















