"""Complexity Economics and Sustainable Development
   Chapter 10 source code
   
   Description: runs algorithm to optimise fedral transfers

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')



 
# Dataset
df = pd.read_csv(home+"/data/chapter_10/indicators.csv")
colYears = [col for col in df.columns if col.isnumeric()]

dfb = pd.read_csv(home+"/data/chapter_10/budgets.csv")
dfb = dfb[dfb.year.isin([int(c) for c in colYears])]
dfb = dfb.groupby('stateAbbreviation').mean()
dfp = pd.read_csv(home+"/data/chapter_10/populations.csv")
pop_dict = dict([(code, dfp[dfp.stateCode==code][colYears[0:-1]].values.mean()) for code in dfb.index.values])
pop_means = [dfp[dfp.stateCode==code][colYears[0:-1]].values.mean() for code in dfb.index.values]
dfb['total_pc'] = dfb['total'].values/pop_means
dfb['participations_pc'] = dfb['participations'].values/pop_means
dfb['contributions_pc'] = dfb['contributions'].values/pop_means

all_sdgs = ['all'] + df.sdg.unique().tolist()

colors = {1:'forestgreen', 2:'darkorange', 3:'gray'}

clusters = dict(zip(df.stateAbbreviation, df.group))
states = sorted(df.stateAbbreviation.unique())


budgets = dict([(state, {}) for state in states])

for state in states:
    
    # Budget 
    total = dfb[dfb.index==state].total.values[0]
    budgets[state]['total'] = total/pop_dict[state]
    
    participations = dfb[dfb.index==state].participations.values[0]
    budgets[state]['participations'] = participations
    
    contributions = dfb[dfb.index==state].contributions.values[0]
    budgets[state]['contributions'] = contributions
    
    transfers = participations + contributions
    budgets[state]['transfers'] = transfers
        
    

stateCluster = dict(zip(df.stateAbbreviation, df.group))

new_rows = []
for sdg in all_sdgs:
    df_opt = pd.read_csv(home+"/data/chapter_10/simulation/optimisation/"+str(sdg)+".csv")  
    for state in states:
        emp_bud = budgets[state]['contributions']#/pop_dict[state]
        opt_bud = df_opt[state].values[0]*dfb['contributions'].sum()#/pop_dict[state]
        
        new_row = ['contribution', sdg, state, stateCluster[state], emp_bud, opt_bud]
        new_rows.append(new_row)


dfn = pd.DataFrame(new_rows, columns=['type', 'sdg', 'state', 'cluster', 'emp_bud', 'opt_bud'])
dfn.to_csv(home+"/data/chapter_10/results/optimal.csv", index=False)
























