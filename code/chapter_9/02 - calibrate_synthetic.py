"""Complexity Economics and Sustainable Development
   Chapter 9 source code
   
   Description: calibrates the model parameters when using data without aid flows

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi






df = pd.read_csv(home+"/data/chapter_9/indicators.csv")
groups_dict = dict(zip(df.countryCode, df.group))
colYears = (np.array([year for year in range(2000, 2014)])).astype(str)
num_years = len(colYears)



countries = df.countryCode.unique()
parallel_processes = 4
sub_periods = 4
T = len(colYears)*sub_periods



for country in countries:

    # Extract country data
    dft = df[df.countryCode==country]
    df_exp = pd.read_csv(home+"/data/chapter_9/budgets/expenditure/"+country+".csv")
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    
    Imax = None
    Imin=None
    R = dft.instrumental.values.copy()
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
            IF.append(np.min(serie[serie!=serie[0]]))
    
    IF = np.array(IF)
    success_rates = ppi.get_success_rates(series)
    mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
    mean_drops[np.isnan(mean_drops)] = 0
    aa = np.abs(mean_drops/sub_periods)
    
    
    # Budget 
    expenditure = np.clip([signal.detrend(serie)+np.mean(serie) for serie in df_exp[colYears].values], a_min=0, a_max=None)
    Bs = expenditure
    usdgs = sorted(dft.sdg.unique())
    sdg2index = dict(zip(usdgs, range(len(usdgs))))
    sdgs = dft.sdg.values
    B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
    Bs = ppi.get_dirsbursement_schedule(Bs, B_dict, T)
    
    # Network
    A = np.loadtxt(home+"/data/chapter_9/networks/"+country+".csv", delimiter=',')
    
    # Governance
    qm = np.ones(n)*dft.controlOfCorruption.values[0]
    rl = np.ones(n)*dft.ruleOfLaw.values[0]
    
    print(country)
    outputs = ppi.calibrate(I0, IF, success_rates, A=A, R=R, qm=qm, rl=rl,  Bs=Bs, B_dict=B_dict, 
              threshold=.9, parallel_processes=parallel_processes, verbose=False, low_precision_counts=101, increment=1000)
    
    dfc = pd.DataFrame(outputs[1::,:].astype(float), columns=outputs[0])
    dfc['years'] = np.ones(N)*np.nan
    dfc.loc[0, 'years'] = num_years
    
    dfc.to_csv(home+'/data/chapter_9/parameters_synthetic/'+country+'.csv', index=False)
    
    
    
    










