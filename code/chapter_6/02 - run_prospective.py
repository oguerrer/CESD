"""Complexity Economics and Sustainable Development
   Chapter 6 source code
   
   Description: performs simulations looking into the future

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')
import ppi



df = pd.read_csv(home+"/data/chapter_6/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")



countries = df.countryCode.unique()
sample_size = 1000
parallel_processes = 4



for country in countries:
    
    print(country)
    
    dft = df[df.countryCode==country]
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
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    R = dft.instrumental.values.copy()
    n = R.sum()
    I0 = dft['2020']
    Imax = np.ones(N)
    Imin = np.zeros(N)
    
    # Budget   
    Bs = np.array([np.ones(years_forward)*df_expt[colYears].values.mean()])
    B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    Bs = ppi.get_dirsbursement_schedule(Bs, B_dict, years_forward*sub_periods)
    
    # Network
    A = np.loadtxt(home+"/data/chapter_6/networks/"+country+".csv", delimiter=',')
    
    # Governance
    qm = np.ones(n)*dft.controlOfCorruption.values[0]
    rl = np.ones(n)*dft.ruleOfLaw.values[0]
    
    # Simulation   
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppi.run_ppi_parallel)\
            (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl, Imax=Imax, 
             Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(sample_size)))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_Is = np.mean(tsI, axis=0)
    
    dff = pd.DataFrame(np.hstack([[[c] for c in dft.seriesCode], aver_Is]), columns=['seriesCode']+list(range(aver_Is.shape[1])))
    dff.to_csv(home+"data/chapter_6/simulation/prospective/"+country+".csv", index=False)

    
    
    
    
    
























    