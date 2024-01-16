"""Complexity Economics and Sustainable Development
   Chapter 7 source code - Figures 7.2, 7.3, 7.4, and 7.5
   Chapter 7 uses the data and calibration of Chapter 6

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
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
    alphas_prime = dfp.alpha_prime.values
    betas = dfp.beta.values
    T = dfp['T'].values[0]
    num_years = dfp.years.values[0]
    sub_periods = int(T/num_years)
    
    x = [[i] for i in range(years_forward*sub_periods)]
    
    
    country = dft.countryCode.values[0]
    group = dft.group.values[0]
    new_row = [[country, group, dft.loc[i, 'sdg'], dft.loc[i, 'seriesCode']] for i in range(len(dft))]
    
    df_prosp = pd.read_csv(home+"/data/chapter_6/simulation/prospective/"+country+".csv")
    series_prosp = df_prosp.values[:,1::]
    
    df_front = pd.read_csv(home+"/data/chapter_7/simulation/frontier/"+country+".csv")
    series_front = df_front.values[:,1::]       

    df_front90 = pd.read_csv(home+"/data/chapter_7/simulation/frontier90/"+country+".csv")
    series_front90 = df_front90.values[:,1::] 

    
    for index, row in dft.iterrows():
        
        goal = row.goal
        group = row.group
        indi = row.seriesCode
        gap0 = 100*(goal - series_prosp[index][0])/goal
        gap = 100*(goal - series_prosp[index][sub_periods*10-1])/goal
        gap_prosp = gap
        if gap_prosp < 0:
            gap_prosp = 0
        if gap_prosp == 0 or gap0 == 0:
            gap_closure_prosp = np.nan
        else:
            gap_closure_prosp = 100*(1-gap/gap0)
        meanLevel = series_prosp[index].mean()
        inivalProp = series_prosp[index][0]
        finvalProp = series_prosp[index][sub_periods*10-1]
        
            
        gap0 = 100*(goal - series_front[index][0])/goal
        gap = 100*(goal - series_front[index][sub_periods*10-1])/goal
        gap_front = gap
        if gap_front < 0:
            gap_front = 0
        if gap_front == 0 or gap0 == 0:
            gap_closure_front = np.nan
        else:
            gap_closure_front = 100*(1-gap/gap0)
        inivalFront = series_front[index][0]
        finvalFront = series_front[index][sub_periods*10-1]
        
        
        gap0 = 100*(goal - series_front90[index][0])/goal
        gap = 100*(goal - series_front90[index][sub_periods*10-1])/goal
        gap_front90 = gap
        if gap_front90 < 0:
            gap_front90 = 0
        if gap_front90 == 0 or gap0 == 0:
            gap_closure_front90 = np.nan
        else:
            gap_closure_front90 = 100*(1-gap/gap0)
        inivalFront90 = series_front90[index][0]
        finvalFront90 = series_front90[index][sub_periods*10-1]
    
        
        distances = np.where( np.abs(series_prosp[index][sub_periods*10-1] <= series_front[index]) )[0]
        savings = 0
        if len(distances) > 0:
            period = distances[0]
            if period < sub_periods*10-1:
                savings = (sub_periods*10-1 - period)/sub_periods

        
        new_rows.append([row.countryCode, row.group, row.sdg, row.seriesCode, row.instrumental, inivalProp, finvalProp, inivalFront90, finvalFront90, inivalFront, finvalFront, meanLevel, gap_prosp, gap_front, gap_front90, savings])






dfn = pd.DataFrame(new_rows, columns=['countryCode', 'group', 'sdg', 'seriesCode', 'instrumental', 'inivalProp', 'finvalProp', 'inivalFront90', 'finvalFront90', 'inivalFront', 'finvalFront', 'meanLevel', 'meanGapProspective', 'meanGapFrontier', 'meanGapFrontier90', 'savings'])
dfn.sort_values(by=['countryCode', 'sdg', 'seriesCode'], inplace=True)
dfn.to_csv(home+"/data/chapter_7/results/frontier_sdr.csv", index=False)


    













