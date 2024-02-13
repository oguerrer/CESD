"""Complexity Economics and Sustainable Development
   Chapter 11 source code - Figures 11.2, 11.3, and 11.4
   
   Description: runs simulations to detect accelerators

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')






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


A = np.loadtxt(home+"/data/chapter_11/network.csv", delimiter=',')






targets = df_exp.target.values

M = np.ones((len(all_targets), len(all_targets))) * np.nan
E = np.ones((len(all_targets), len(all_targets))) * np.nan
impacts = {}



all_impacts_bottlenecks = {}
all_impacts_accelerators = {}
for target_intervened in targets:
    print(target_intervened)
    
    dft = pd.read_csv(home+"/data/chapter_11/simulation/bottlenecks/"+target_intervened+'.gzip', compression='gzip',
                      dtype={'target_intervened':str, 'target_affected1':str, 'target_affected2':str})    
    impacts = []
    for trial, group in dft.groupby('trial'):
        subgroup = group[(group.target_affected1 != target_intervened) & (group.target_affected2 != target_intervened)]
        impact = 100*subgroup.area_counter.values.sum()/subgroup.area_baseline.values.sum()
        impacts.append(impact)
    all_impacts_bottlenecks[target_intervened] = impacts
    
    dft = pd.read_csv(home+"/data/chapter_11/simulation/accelerators/"+target_intervened+'.gzip', compression='gzip',
                      dtype={'target_intervened':str, 'target_affected1':str, 'target_affected2':str})    
    impacts = []
    for trial, group in dft.groupby('trial'):
        subgroup = group[(group.target_affected1 != target_intervened) & (group.target_affected2 != target_intervened)]
        impact = -100*subgroup.area_counter.values.sum()/subgroup.area_baseline.values.sum()
        impacts.append(impact)
    all_impacts_accelerators[target_intervened] = impacts
    




statistic_bottlenecks = {}
mean_bottlenecks = {}
for target_intervened, impacts in all_impacts_bottlenecks.items():
    statistic = np.percentile(impacts, 10)
    statistic_bottlenecks[target_intervened] = statistic
    mean_bottlenecks[target_intervened] = np.mean(impacts)




statistic_accelerators = {}
mean_accelerators = {}
for target_intervened, impacts in all_impacts_accelerators.items():
    statistic = np.percentile(impacts, 10)
    statistic_accelerators[target_intervened] = statistic
    mean_accelerators[target_intervened] = np.mean(impacts)
    
    
    
budget_dict = dict(zip(df_exp.target, df_exp[colYears].values.mean(axis=1)))

spillovers = {}
for target in targets:
    rows = A[(df.target==target) | (df.target2==target)]
    spillovers[target] = rows[:,(df.target!=target) & (df.target2!=target)].mean()


data = [[target_intervened, statistic_bottlenecks[target_intervened], mean_bottlenecks[target_intervened],
          statistic_accelerators[target_intervened], mean_accelerators[target_intervened], budget_dict[target_intervened], 
          spillovers[target_intervened]] for target_intervened in targets]

dfn = pd.DataFrame(data, columns=['target_intervened', 'statistic_bottlenecks', 'mean_bottlenecks', 
                                  'statistic_accelerators', 'mean_accelerators', 'budget', 'spillovers'])
dfn.to_csv(home+"/data/chapter_11/results/aggregate.csv", index=False)











