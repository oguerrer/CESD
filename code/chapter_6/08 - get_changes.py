"""Complexity Economics and Sustainable Development
   Chapter 6 source code - Figure 6.4
   
   Description: computes the results of alternative amounts of government
       expenditure

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import matplotlib.pyplot as plt
import os, sys, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
    
    dfr = pd.read_csv(home+"/data/chapter_6/simulation/changes/"+country+'_'+"0.csv")
    vals_2030 = dfr.values[:,1::][:,10*sub_periods]
    
    for change in changes:
        dfi = pd.read_csv(home+"/data/chapter_6/simulation/changes/"+country+'_'+str(change)+".csv")
        series = dfi.values[:,1::]
        
        for index, serie in enumerate(series):
            hundreds = np.where(serie==100)[0]
            if len(hundreds) > 0:
                serie = serie[0:hundreds[0]+1]
            
            Y = serie
            X = x[0:len(serie)]
            
            model = LinearRegression()
            model.fit(X, Y)
            
            goal = dft.loc[index, 'goal']
            y_pred = (goal - model.intercept_)/model.coef_[0]
            new_row[index].append( y_pred/sub_periods )
            new_row[index].append( model.coef_[0] )
            
    new_rows += new_row





cCols = np.array([(c, str(c)+'_slope') for c in changes]).flatten().tolist()
dfn = pd.DataFrame(new_rows, columns=['countryCode', 'group', 'sdg', 'seriesCode']+cCols)
dfn.sort_values(by=['countryCode', 'sdg', 'seriesCode'], inplace=True)
dfn.to_csv(home+"/data/chapter_6/results/changes.csv", index=False)





















