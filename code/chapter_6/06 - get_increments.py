"""Complexity Economics and Sustainable Development
   Chapter 6 source code - Figure 6.3

   Description: computes the results of an increased budget

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import scipy.stats as st
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi


df = pd.read_csv(home+"/data/chapter_6/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")

changes = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]



countries = df.countryCode.unique()

new_rows = []
for country in countries:
    
    print(country)
    
    dft = df[df.countryCode==country].copy()
    dft.reset_index(drop=True, inplace=True)
    df_expt = df_exp[df_exp.countryCode==country]
    budget_mean = df_expt[colYears].values.mean()
    budget_change = 100/budget_mean
    
    # Periods forward
    years_forward = 20
    
    # Parameters
    dfp = pd.read_csv(home+'/data/chapter_6/parameters/'+country+'.csv')
    alphas = dfp.alpha.values
    betas = dfp.beta.values
    T = dfp['T'].values[0]
    num_years = dfp.years.values[0]
    sub_periods = int(T/num_years)
    
    x = [[i] for i in range(years_forward*sub_periods)]
    
    
    country = dft.countryCode.values[0]
    group = dft.group.values[0]
    new_row = [[country, group, dft.loc[i, 'sdg'], dft.loc[i, 'seriesCode']] for i in range(len(dft))]
    
    
    df100 = pd.read_csv(home+"/data/chapter_6/simulation/increment/"+country+".csv")
    series100 = df100.values[:,1::]
        
    dfi = pd.read_csv(home+"/data/chapter_6/simulation/prospective/"+country+".csv")
    series = dfi.values[:,1::]
    
    for index, row in dft.iterrows():
        
        goal = row.goal
        group = row.group
        indi = row.seriesCode
        gap0 = 100*(goal - series[index][0])/goal
        gap = 100*(goal - series[index][sub_periods*10-1])/goal
        if gap < 0:
            gap = 0
        if gap0 == 0:
            gap_closure = np.nan
        else:
            gap_closure = 100*(1-gap/gap0)
        gap_closure_original = gap_closure
        gap_final_original = gap
        
            
        goal = row.goal
        group = row.group
        indi = row.seriesCode
        gap0 = 100*(goal - series100[index][0])/goal
        gap = 100*(goal - series100[index][sub_periods*10-1])/goal
        if gap < 0:
            gap = 0
        if gap0 == 0:
            gap_closure100 = np.nan
        else:
            gap_closure100 = 100*(1-gap/gap0)
        gap_final_counter = gap
        
        gap_closure_change = 100*(gap_closure100 - gap_closure) /  gap_closure
        
        if gap_final_original > 0:
            gap_closure_elasticity = ((gap_closure100 - gap_closure)/ gap_closure) / budget_change
        else:
            gap_closure_elasticity = np.nan
        
        new_rows.append([row.countryCode, row.group, row.sdg, row.seriesCode, gap_closure_original, gap_closure_change, gap_closure_elasticity, row.instrumental])






dfn = pd.DataFrame(new_rows, columns=['countryCode', 'group', 'sdg', 'seriesCode', 'meanGap', 'gapClosureChange', 'gapClosureElasticity', 'instrumental'])
dfn.sort_values(by=['countryCode', 'sdg', 'seriesCode'], inplace=True)
dfn.to_csv(home+"/data/chapter_6/results/increment.csv", index=False)

















