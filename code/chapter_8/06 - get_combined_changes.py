"""Complexity Economics and Sustainable Development
   Chapter 8 source code - Figures 8.3 and 8.4
   Chapter 8 uses its own development indicators file and
       the same expenditure file as chapter 6

   Description: produces the data to analyse the response of corruption from
       combined changes in relative and total government expenditure

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import scipy.stats as st
from csaps import csaps
from sklearn.impute import KNNImputer
from csaps import NdGridCubicSmoothingSpline
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi


# Dataset
df = pd.read_csv(home+"/data/chapter_8/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")



countries = df.countryCode.unique()

new_rows = []

for country in countries:

    print(country)    
    dft = df[df.countryCode==country]

    dfG = pd.read_csv(home+'/data/chapter_8/simulation/combined_changes/'+country+'.zip', compression='zip')
    region = dft.group.values[0]
    development = dft[dft.seriesCode=='sdg16_ruleoflaw'][colYears].values.mean()
    expenditure = df_exp[df_exp.countryCode==country][colYears].values.mean()
    
    n = int(len(dfG.deltaB.unique())/1)
    
    X, Y, Z = [], [], []
    
    bins = np.linspace(0, .1, n+1)
    mid_points = [bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
    for B, group in dfG.groupby('deltaB'):
        statistic, bin_edges, binnumber = st.binned_statistic(group.P, group.DB, bins=bins)
        x = mid_points
        y = statistic
        Y += x
        Z += y.tolist()
        X += [B for i in range(len(x))]
        
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    xdata = [sorted(np.unique(Y)), sorted(np.unique(X))]
    XM, YM = np.meshgrid(*xdata, indexing='ij')
    ZM = XM.copy()
    
    for i in range(ZM.shape[0]):
        for j in range(ZM.shape[0]):
            y = XM[i,j]
            x = YM[i,j]
            index = np.where( (X==x) & (Y==y) )[0][0]
            ZM[i,j] = Z[index]
    
    # impute missing values using the K-nearest neighbours algorithm
    ZMt = ZM.copy()
    if np.sum(np.isnan(ZM)) > 0:
        imputer = KNNImputer(n_neighbors=1)
        ZM = imputer.fit_transform(ZM)
    
    ydata = ZM
    ydata_s = csaps(xdata, ydata, xdata, smooth=1.)
    ydata_s[np.isnan(ZMt)] = np.nan
    
    # calculate the roughness metric with the option of leaving out the volatile/imputed data points
    skip = 2
    spline = NdGridCubicSmoothingSpline(np.array(xdata)[:,skip::].tolist(), ydata[skip::,skip::], smooth=1.)
    evalu = spline.__call__(np.array(xdata)[:,skip::].tolist(), nu=[2,2])
    roughness = np.sum( evalu**2 )
    print(roughness)
    
    x_flat = XM.flatten()
    y_flat = YM.flatten()
    z_flat = ydata_s.flatten()
    
    for i in range(len(x_flat)):
        if i==0:
            new_rows.append([ country, region, y_flat[i], x_flat[i], z_flat[i], roughness, development, expenditure ])
        else:
            new_rows.append([ country, region, y_flat[i], x_flat[i], z_flat[i], np.nan, np.nan, np.nan ])
    


dfn = pd.DataFrame(new_rows, columns=[ 'countryCode', 'group',  'gov_exp', 
                                      'rol_exp', 'corruption', 'roughness', 
                                      'development', 'expenditure' ])
dfn.to_csv(home+"/data/chapter_8/results/combined_changes.csv", index=False)





