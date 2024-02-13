"""Complexity Economics and Sustainable Development
   Chapter 10 source code
   
   Description: runs simulations on impacts

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
pop_means = [dfp[dfp.stateCode==code][colYears[0:-1]].values.mean() for code in dfb.index.values]
dfb['total'] = dfb['total'].values/pop_means
dfb['participations'] = dfb['participations'].values/pop_means
dfb['contributions'] = dfb['contributions'].values/pop_means

sample_size = 1000
parallel_processes = 45

sub_periods = 4
T = num_years*sub_periods

states = df.stateAbbreviation.unique()

for state in states:   
    
    dft = df[df.stateAbbreviation == state]
    stateCode = dft.stateCode.values[0]
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    
    R = (dft.instrumental.values == 'I').astype(int)
    n = R.sum()
    I0 = series[:,0]
    Imax = dft.Imax.values
    success_rates = ppi.get_success_rates(series)

    
    # Budget 
    B0s = np.array([np.ones(num_years)*dfb[dfb.index==state].total.values[0]])
    B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    contributions = dfb[dfb.index==state].contributions.values[0]
    participations = dfb[dfb.index==state].participations.values[0]
    transfers = participations + contributions
    cont_proportions = np.clip(contributions/B0s, 0, 1)
    cont_transfers = np.clip(transfers/B0s, 0, 1)
    counter_budgets = {'contributions':B0s*(1-cont_proportions), 'transfers':B0s*(1-cont_transfers)}
    
    # Network
    A = np.loadtxt(home+"/data/chapter_10/networks/"+state+".csv", delimiter=',')
    
    # Governance
    qm = dft.monitoring.values[R==1]
    rl = dft.ruleOfLaw.values[R==1]
    
    
    # Parameters
    dfp = pd.read_csv(home+"/data/chapter_10/parameters/"+state+".csv")
    alphas = dfp.alpha.values
    alphas_prime = dfp.alpha_prime.values
    betas = dfp.beta.values
    
    
    print(state)
    # baseline
    B0s = ppi.get_dirsbursement_schedule(B0s, B_dict, T)
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppi.run_ppi_parallel)\
            (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
             Bs=B0s, B_dict=B_dict, Imax=Imax) for itera in range(sample_size)))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_indis_0 = np.mean(tsI, axis=0)
    
    
    # budget minus federal transfers
    aver_indis_c = []
    for type_transfer in ['transfers', 'contributions']:
        B0x = ppi.get_dirsbursement_schedule(counter_budgets[type_transfer], B_dict, T)
        sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppi.run_ppi_parallel)\
                (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
                 Bs=B0x, B_dict=B_dict, Imax=Imax) for itera in range(sample_size)))
        tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
        aver_indis_c.append(np.mean(tsI, axis=0))

    
    M = np.column_stack( [dft.seriesCode.values, aver_indis_0, aver_indis_c[0], aver_indis_c[1]] )
    df_effects = pd.DataFrame(M, columns=['seriesCode'] + ['baseline_'+str(c) for c in range(T)] + 
                              ['counter_transfers_'+str(c) for c in range(T)] + 
                              ['counter_contributions_'+str(c) for c in range(T)])
    df_effects.to_csv(home+'/data/chapter_10/simulation/impact/'+state+'.csv', index=False)
    
    
    
    




