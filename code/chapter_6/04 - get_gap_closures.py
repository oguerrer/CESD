"""Complexity Economics and Sustainable Development
   Chapter 6 source code - Figure 6.2

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import matplotlib.pyplot as plt
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi



df = pd.read_csv(home+"/data/chapter_6/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")



countries = df.countryCode.unique()

data = dict([(group, dict([(indi, []) for indi in df.seriesCode.unique()])) for group in df.group.unique()])

for country in countries:
    
    print(country)
    
    dft = df[df.countryCode==country].copy()
    dft.reset_index(drop=True, inplace=True)
    df_expt = df_exp[df_exp.countryCode==country]
    
    # Periods forward
    years_forward = 20

    # Parameters
    dfp = pd.read_csv(home+'/data/chapter_6/parameters/'+country+'.csv')
    alphas = dfp.alpha.values
    alphas_prime = dfp.alpha_prime.values
    betas = dfp.beta.values
    T = dfp['T'].values[0]
    num_years = dfp.years.values[0]
    sub_periods = int(T/num_years)
        
    # Simulation
    dfi = pd.read_csv(home+"/data/chapter_6/simulation/prospective/"+country+".csv")
    series = dfi.values[:,1::]
    
    for index, row in dft.iterrows():
        
        goal = row.goal
        group = row.group
        indi = row.seriesCode
        gap0 = 100*(goal - series[index][0])/goal
        gap = 100*(goal - series[index][sub_periods*10-1])/goal
        if gap0 <= 0:
            gap_closure = 0
        else:
            gap_closure = 100*(gap0-gap)/gap0
        
        data[group][indi].append(gap_closure)



new_rows = []
for group in df.group.unique():
    for indi in df.seriesCode.unique():
        new_row = [group, int(indi.split('_')[0][3::]), indi, np.nanmean(data[group][indi])]
        new_rows.append(new_row)


dfn = pd.DataFrame(new_rows, columns=['group', 'sdg', 'seriesCode', 'meanGapClosure',])
dfn.sort_values(by=['group', 'sdg', 'seriesCode'], inplace=True)
dfn.to_csv(home+"/data/chapter_6/results/gap_closures.csv", index=False)








