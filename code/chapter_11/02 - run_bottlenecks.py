"""Complexity Economics and Sustainable Development
   Chapter 11 source code
   
   Description: runs simulations to detect systemic bottlenecks

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import powerlaw
from sklearn.linear_model import LinearRegression
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')
import ppi





# Dataset
df = pd.read_csv(home+'/data/chapter_11/indicators.csv')
df_exp = pd.read_csv(home+"/data/chapter_11/budget.csv")
df_exp = df_exp[df_exp.target.isin(df.target.values.tolist()+df.target2.values.tolist())]
all_targets = sorted(list(set(df.target[~df.target.isnull()].values.tolist() + 
                              df.target2[~df.target2.isnull()].values.tolist())), 
                     key=lambda x: (int(x.split('.')[0]), len(x.split('.')[1]) ,x.split('.')[1]))
targets2ind = dict([(target, i) for i, target in enumerate(all_targets)])
ind2target = dict([(i, target) for i, target in enumerate(all_targets)])
colYears = [col for col in df_exp.columns if col.isnumeric()]

num_years = len(colYears)

sub_periods = 4
T = len(colYears)*sub_periods
parallel_processes = 4
sample_size = 100
sub_sample_size = 1000

series = df[colYears].values
N = len(df)

Imax = np.ones(N)
Imin = np.zeros(N)
R = np.ones(N)
n = R.sum()
I0 = series[:,0]
IF = []
for serie in series:
    x = np.array([float(year) for year in colYears]).reshape((-1, 1))
    y = serie
    model = LinearRegression().fit(x, y)
    coef = model.coef_
    if coef > 0 and serie[-1] > serie[0]:
        IF.append(serie[-1])
    elif coef > 0 and serie[-1] <= serie[0]:
        IF.append(np.max(serie[serie!=serie[0]]))
    elif coef < 0 and serie[-1] < serie[0]:
        IF.append(serie[-1])
    elif coef < 0 and serie[-1] >= serie[0]:
        IF.append(np.min(serie[serie!=serie[0]]))

IF = np.array(IF)
success_rates = ppi.get_success_rates(series)
mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
mean_drops[np.isnan(mean_drops)] = 0
aa = np.abs(mean_drops/sub_periods)

# Network
A = np.loadtxt(home+"/data/chapter_11/network.csv", delimiter=',')

# Governance
qm = df.cc.values.copy()
rl = df.rl.values.copy()

# Budget 
Bs = df_exp[colYears].values.mean(axis=1)
Bs = np.tile(Bs, (num_years,1)).T
B_dict = {}
for index, row in df.iterrows():
    B_dict[index] = []
    for target in df_exp.target.values:
        if row.target == target or row.target2 == target:
            B_dict[index].append(targets2ind[target])
Bs = ppi.get_dirsbursement_schedule(Bs, B_dict, T)

# Parameters
dfp = pd.read_csv(home+'/data/chapter_11/parameters.csv')
alphas = dfp.alpha.values
alphas_prime = dfp.alpha_prime.values
betas = dfp.beta.values

variances = []
for values in df_exp[colYears].values:
    y = (values[1::] - values[0:-1])/values[0:-1]
    y = y[y>0]
    variances += y.tolist()

variances = np.abs(variances)
variances = variances[(~np.isnan(variances)) & (~np.isinf(variances))]
a,b,c = powerlaw.fit(variances)


# Baseline
sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppi.run_ppi)\
                (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
                 Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(1000)))
tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
I_hat = np.mean(tsI, axis=0)
    


# Counterfactuals

targets = df_exp.target.values

for ti, target_intervened in enumerate(targets):
    
    outputs = []
    for trial in range(sub_sample_size):
        print(target_intervened, trial)
        budget_change = np.random.rand() #powerlaw.rvs(a)
        Bsc = Bs.copy()
        Bsc[ti] *= np.clip(1 - budget_change, a_min=0, a_max=None) 

        
        sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppi.run_ppi)\
                        (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
                         Imax=Imax, Imin=Imin, Bs=Bsc, B_dict=B_dict) for itera in range(sample_size)))
        tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
        I_hat_c = np.mean(tsI, axis=0)
        
        
        for index, row in df.iterrows():
            area_baseline = np.sum(I_hat[index] - I_hat[index].min())
            area_counter = np.sum(I_hat[index] - I_hat_c[index])
            budget_baseline = Bs[B_dict[index]]
            budget_counter = Bsc[B_dict[index]]
            outputs.append([trial, row.seriesCode, target_intervened, row.target, row.target2, 
                            area_baseline, area_counter, budget_baseline, budget_counter, budget_change])

    
    dff = pd.DataFrame(outputs, columns=['trial', 'series_code', 'target_intervened', 'target_affected1', 'target_affected2', 
                                         'area_baseline', 'area_counter', 'budget_baseline', 'budget_counter', 'budget_change'])
    dff.to_csv(home+'/data/chapter_11/simulation/bottlenecks/'+target_intervened+'.gzip', 
                index=False, compression={'method': 'gzip', 'compresslevel': 9})
    

























