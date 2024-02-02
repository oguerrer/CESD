"""Complexity Economics and Sustainable Development
   Chapter 8 source code - Figure 8.2
   Chapter 8 uses its own development indicators file and
       the same expenditure file as chapter 6

   Description: produces the data to analyse the response of corruption as a
       function of changes in relative and total government expenditure

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.interpolate import UnivariateSpline
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi


# Dataset
df = pd.read_csv(home+"/data/chapter_8/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")






new_rows = []

for country, dft in df.groupby('countryCode'):
    
    print(country)
    
    dfB = pd.read_csv(home+'/data/chapter_8/simulation/total_changes/'+country+'.zip', compression='zip')

    grouped = dfB.groupby('deltaB').mean()

    xB = grouped.index.values
    yB = grouped.DB.values
    f = UnivariateSpline(xB, yB, k=3)
    y_splineB = f(xB)
    
    R = (dft.instrumental.values == 1).astype(int)
    rl = int(np.where(dft.seriesCode.values == 'sdg16_ruleoflaw')[0][0])
    rl_sub_index = np.where(np.where(R==1)[0]==rl)[0][0]
    
    dfP = pd.read_csv(home+'/data/chapter_8/simulation/relative_changes/'+country+'.zip', compression='zip')
    bins = np.linspace(0, .1, 101)
    mid_points = np.array([bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)])
    statistic, bin_edges, binnumber = st.binned_statistic(dfP['indi_P_'+str(rl_sub_index)], dfP.DB, bins=bins)
    
    xbrlt = mid_points[~np.isnan(statistic)]
    ybrlt = statistic[~np.isnan(statistic)]
    f = UnivariateSpline(xbrlt, ybrlt, k=3)
    y_splinebrlt = f(xbrlt)
    
    n = len(xB)
    xbrl = mid_points
    ybrl = statistic
    y_splinebrl = np.ones(n)*np.nan
    y_splinebrl[~np.isnan(statistic)] = y_splinebrlt
    
    for i in range(n):
        new_rows.append( [country, dft.group.values[0], xB[i], yB[i], y_splineB[i], xbrl[i], ybrl[i], y_splinebrl[i]] )






dfn = pd.DataFrame(new_rows, columns=['countryCode', 'group', 'exp_B', 'corr_B', 'spline_B', 'exp_brl', 'corr_brl', 'spline_brl'])
dfn.to_csv(home+"/data/chapter_8/results/relative_and_total_changes.csv", index=False)


















