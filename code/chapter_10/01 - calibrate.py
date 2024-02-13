"""Complexity Economics and Sustainable Development
   Chapter 10 source code
   
   Description: calibrates the model parameters

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')
import ppi





 
# Dataset
df = pd.read_csv(home+"/data/chapter_10/indicators.csv")
colYears = [col for col in df.columns if col.isnumeric()]
num_years = len(colYears)

dfb = pd.read_csv(home+"/data/chapter_10/budgets.csv")
dfb = dfb[dfb.year.isin([int(c) for c in colYears])]
dfb = dfb.groupby('stateAbbreviation').mean()
dfp = pd.read_csv(home+"/data/chapter_10/populations.csv")
pop_means = [dfp[dfp.stateCode==code][colYears[0:-1]].values.mean() for code in dfb.index.values]
dfb['total'] = dfb['total'].values/pop_means
dfb['participations'] = dfb['participations'].values/pop_means
dfb['contributions'] = dfb['contributions'].values/pop_means


parallel_processes = 4
states = df.stateAbbreviation.unique()
sub_periods = 4
T = len(colYears)*sub_periods



for state in states:   
    
    dft = df[df.stateAbbreviation == state]
    stateCode = dft.stateCode.values[0]
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    
    R = (dft.instrumental.values == 'I').astype(int)
    n = R.sum()
    I0 = series[:,0]
    IF = []
    for serie in series:
        x = np.array([float(year) for year in colYears]).reshape((-1, 1))
        y = serie
        model = LinearRegression().fit(x, y)
        coef = model.coef_
        if coef > 0 and serie[-1] > serie[0]:
            IF.append(serie[-1])
        elif coef > 0 and serie[-1] <= serie[0]:
            IF.append(np.max(serie[serie!=serie[0]]))
        elif coef < 0 and serie[-1] < serie[0]:
            IF.append(serie[-1])
        elif coef < 0 and serie[-1] >= serie[0]:
            IF.append(np.min(serie))
        else:
            IF.append(serie[-1])
    
    IF = np.array(IF)
    IF[(I0==IF)] = IF[(I0==IF)] + 1e-6
    
    success_rates = ppi.get_success_rates(series)
    mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
    mean_drops[np.isnan(mean_drops)] = 0
    aa = np.abs(mean_drops/sub_periods)
    
    
    # Budget 
    B0s = np.array([np.ones(len(colYears))*dfb[dfb.index==state].total.values[0]])
    B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    
    # Network
    A = np.loadtxt(home+"/data/chapter_10/networks/"+state+".csv", delimiter=',')
    
    # Governance
    qm = dft.monitoring.values[R==1]
    rl = dft.ruleOfLaw.values[R==1]
    
    
    print(state)
    Bs = ppi.get_dirsbursement_schedule(B0s, B_dict, T)
    outputs = ppi.calibrate(I0, IF, success_rates, A=A, R=R, qm=qm, rl=rl,  Bs=Bs, B_dict=B_dict,
              threshold=.9, parallel_processes=parallel_processes, verbose=True, low_precision_counts=101, increment=1000)
    
    dfc = pd.DataFrame(outputs[1::,:].astype(float), columns=outputs[0])
    dfc['years'] = np.ones(N)*np.nan
    dfc.loc[0, 'years'] = num_years
    
    dfc.to_csv(home+'/data/chapter_10/parameters/'+state+'.csv', index=False)
    
    




