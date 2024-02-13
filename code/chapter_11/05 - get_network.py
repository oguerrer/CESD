"""Complexity Economics and Sustainable Development
   Chapter 11 source code - Figure 11.5
   
   Description: runs simulations to detect accelerators

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as st
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




GA = nx.DiGraph()
GB = nx.DiGraph()

GA.add_nodes_from(targets, important=False)
GB.add_nodes_from(targets, important=False)

for target_intervened in targets:
    print(target_intervened)
        
    # BOTTLENECKS
    dft = pd.read_csv(home+"/data/chapter_11/simulation/bottlenecks/"+target_intervened+'.gzip', compression='gzip',
                      dtype={'target_intervened':str, 'target_affected1':str, 'target_affected2':str})    
    
    impacts = []
    for trial, group in dft.groupby('trial'):
        subgroup = group[(group.target_affected1 != target_intervened) & (group.target_affected2 != target_intervened)]
        impact = 100*subgroup.area_counter.values.sum()/subgroup.area_baseline.values.sum()
        impacts.append(impact)
    statistic = np.percentile(impacts, 10)
    if statistic > 0:
        GB.nodes[target_intervened]['important'] = True
        dfg = dft[(dft.target_affected1 != target_intervened) & (dft.target_affected2 != target_intervened)].groupby(['target_affected1', 'trial']).sum()
        dfg['impact'] = 100*dfg.area_counter/dfg.area_baseline
        for target_affected, group in dfg.groupby('target_affected1'):
            statistic2, pvalue = st.ttest_1samp(group.impact.values, 0)
            mean = np.mean(group.impact.values)
            if statistic2 > 0 and pvalue < .01:
                GB.add_edge(target_intervened, target_affected, weight=mean)

        dfg = dft[(dft.target_affected1 != target_intervened) & (dft.target_affected2 != target_intervened)].groupby(['target_affected2', 'trial']).sum()
        dfg['impact'] = 100*dfg.area_counter/dfg.area_baseline
        for target_affected, group in dfg.groupby('target_affected2'):
            statistic2, pvalue = st.ttest_1samp(group.impact.values, 0)
            mean = np.mean(group.impact.values)
            if statistic2 > 0 and pvalue < .01:
                GB.add_edge(target_intervened, target_affected, weight=mean)
    
    
    # ACCELERATORS
    dft = pd.read_csv(home+"/data/chapter_11/simulation/accelerators/"+target_intervened+'.gzip', compression='gzip',
                      dtype={'target_intervened':str, 'target_affected1':str, 'target_affected2':str})    
    
    impacts = []
    for trial, group in dft.groupby('trial'):
        subgroup = group[(group.target_affected1 != target_intervened) & (group.target_affected2 != target_intervened)]
        impact = -100*subgroup.area_counter.values.sum()/subgroup.area_baseline.values.sum()
        impacts.append(impact)
    statistic = np.percentile(impacts, 10)
    if statistic > 0:
        GA.nodes[target_intervened]['important'] = True
        dfg = dft[(dft.target_affected1 != target_intervened) & (dft.target_affected2 != target_intervened)].groupby(['target_affected1', 'trial']).sum()
        dfg['impact'] = 100*dfg.area_counter/dfg.area_baseline
        for target_affected, group in dfg.groupby('target_affected1'):
            statistic2, pvalue = st.ttest_1samp(group.impact.values, 0)
            mean = np.mean(group.impact.values)
            if statistic2 > 0 and pvalue < .01:
                GA.add_edge(target_intervened, target_affected, weight=mean)
    
        dfg = dft[(dft.target_affected1 != target_intervened) & (dft.target_affected2 != target_intervened)].groupby(['target_affected2', 'trial']).sum()
        dfg['impact'] = 100*dfg.area_counter/dfg.area_baseline
        for target_affected, group in dfg.groupby('target_affected2'):
            statistic2, pvalue = st.ttest_1samp(group.impact.values, 0)
            mean = np.mean(group.impact.values)
            if statistic2 > 0 and pvalue < .01:
                GA.add_edge(target_intervened, target_affected, weight=mean)


    


dfn = nx.to_pandas_edgelist(GB)
dfn.to_csv(home+"/data/chapter_11/results/network_bottlenecks.csv", index=False)

dfn = nx.to_pandas_edgelist(GA)
dfn.to_csv(home+"/data/chapter_11/results/network_accelerators.csv", index=False)




