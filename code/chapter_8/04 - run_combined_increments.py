"""Complexity Economics and Sustainable Development
   Chapter 8 source code
   Chapter 8 uses its own development indicators file and
       the same expenditure file as chapter 6

   Description: simulates combinations of relative and total changes in
       government expenditure

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





# We define a function that returns various averages from the PPI outputs
def run_ppi_bs(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None,
            Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=50, frontier=None, rl_sub_index=None):
    
    outputs = ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, bs=bs, qm=qm, rl=rl,
            Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict, G=G, T=T, frontier=frontier)
    
    tsI, tsC, tsF, tsP, tsS, tsG = outputs
    tsD = tsP-tsC
    
    DB = tsD.sum()/tsP.sum()
    P = tsP[rl_sub_index].sum()/tsP.sum()
    return (DB, P)





# Dataset
df = pd.read_csv(home+"/data/chapter_8/indicators.csv")
colYears = [col for col in df.columns if str(col).isnumeric() and int(col)<2021]
df_exp = pd.read_csv(home+"/data/chapter_6/expenditure.csv")


countries = df.countryCode.unique()
sample_size = 1000
parallel_processes = 4



for country in countries:
        
    # Extract country data
    dft = df[df.countryCode==country]
    df_expt = df_exp[df_exp.countryCode==country]
    
    # Parameters
    dfp = pd.read_csv(home+'/data/chapter_8/parameters/'+country+'.csv')
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
    Imax = np.ones(N)
    Imin = np.zeros(N)
    I0 = series[:,0]
    
    # Budget
    Bs = np.array([np.ones(int(num_years))*df_expt[colYears].values.mean()])
    B_dict = dict([(i,[0]) for i in range(N) if R[i]==1])
    Bs = ppi.get_dirsbursement_schedule(Bs, B_dict, num_years*sub_periods)
    
    # Network
    A = np.loadtxt(home+"/data/chapter_8/networks/"+country+".csv", delimiter=',')

    # Governance
    qm = int(np.where(dft.seriesCode.values == 'sdg16_contcorr')[0][0])
    rl = int(np.where(dft.seriesCode.values == 'sdg16_ruleoflaw')[0][0])
    
    bs = np.ones(n)
    B_n0 = Bs.copy()
    rl_sub_index = np.where(np.where(R==1)[0]==rl)[0][0]
    
    deltaB_range = np.linspace(0, 1, 25)
    bs_range = 1-(1-(np.logspace(1, 0, 25)-1)/9)*.5

    all_DB = []
    all_P = []
    all_Bs = []
    all_brl = []
    
    for deltaB in deltaB_range:

        B_n = B_n0*(1+deltaB)
        
        for b_rl in bs_range: 
            
            print(country, deltaB, b_rl)
            
            bs[rl_sub_index] = b_rl
        
            outputs = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_bs)\
            (I0=I0, alphas=alphas, alphas_prime=alphas_prime, betas=betas, A=A, R=R, bs=bs,
             qm=qm, rl=rl, Imax=Imax, Imin=Imin, Bs=B_n, B_dict=B_dict, rl_sub_index=rl_sub_index) for itera in range(sample_size))
            
                
            values = np.array(outputs)
            all_DB += values[:,0].tolist()
            all_P += values[:,1].tolist()
            all_Bs += (np.ones(values.shape[0])*deltaB).tolist()
            all_brl += (np.ones(values.shape[0])*b_rl).tolist()
    
    df_data = pd.DataFrame( np.array([all_DB, all_P, all_Bs, all_brl]).T , 
                           columns=['DB', 'P', 'deltaB', 'brl'])
    df_data = df_data.round(decimals=5)
    df_data.to_csv(home+'/data/chapter_8/simulation/combined_changes/'+country+'.zip', index=False, compression='zip')




