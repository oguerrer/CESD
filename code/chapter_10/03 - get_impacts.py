"""Complexity Economics and Sustainable Development
   Chapter 10 source code - Fugure 10.2
   
   Description: produces the data

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')







df = pd.read_csv(home+"/data/chapter_10/indicators.csv")
colYears = [col for col in df.columns if col.isnumeric()]

states = df.stateAbbreviation.unique()

type_transfer = 'contributions'


area_baseline_group_sdgs = dict([(group, dict([(sdg, 0) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])
area_counter_group_sdgs = dict([(group, dict([(sdg, 0) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])

for state in states:
    
    print(state)
    
    dft = df[df.stateAbbreviation==state]
    R = (dft.instrumental.values=='I').astype(int)
    dft = dft[R==1]
    df_effects = pd.read_csv(home+'/data/chapter_10/simulation/impact/'+state+'.csv')
 
    baselines = [c for c in df_effects.columns if 'baseline' in c]
    counters = [c for c in df_effects.columns if 'counter_'+type_transfer in c]
    
    series_baseline = df_effects[baselines].values
    series_counter = df_effects[counters].values
    area_baseline = np.sum([serie - serie.min() for serie in series_baseline], axis=1)
    area_counter = np.sum([series_baseline[i] - serie_c for i, serie_c in enumerate(series_counter)], axis=1)
    
    i = 0
    for index, row in dft.iterrows():
        area_baseline_group_sdgs[row.group][row.sdg] += area_baseline[i]
        area_counter_group_sdgs[row.group][row.sdg] += area_counter[i]
        i+=1
    
    
new_rows = []
for group in area_baseline_group_sdgs.keys():
    for sdg in area_baseline_group_sdgs[group].keys():
        if area_baseline_group_sdgs[group][sdg] > 0:
            statistic = 100*area_counter_group_sdgs[group][sdg]/area_baseline_group_sdgs[group][sdg]
        else:
            statistic = np.nan
        new_row = [group, sdg, statistic]
        new_rows.append(new_row)

        
dfn = pd.DataFrame(new_rows, columns=['group', 'sdg', 'statistic'])
        
dfn.to_csv(home+"/data/chapter_10/results/"+type_transfer+".csv", index=False)






