"""Complexity Economics and Sustainable Development
   Chapter 8 source code
   Chapter 8 uses its own development indicators file and
       the same expenditure file as chapter 6

   Description: simulates relative expenditure increments directed to 
       the rule of law

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
            Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=50, frontier=None):
    
    outputs = ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, bs=bs, qm=qm, rl=rl,
            Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict, G=G, T=T, frontier=frontier)
    
    tsI, tsC, tsF, tsP, tsS, tsG = outputs
    tsD = tsP-tsC
    B = np.tile(Bs.sum(axis=0), (tsP.shape[0],1))
    
    aver_DB = tsD.sum()/tsP.sum()
    aver_DP = (tsD/tsP).mean(axis=1)
    aver_P = (tsP/B).mean(axis=1)
    aver_rl = tsI[rl].mean()
    aver_qm = tsI[qm].mean()
    return (aver_DB, aver_DP, aver_P, aver_rl, aver_qm)






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
    
    rl_sub_index = np.where(np.where(R==1)[0]==rl)[0][0]
    bs = np.ones(n)
    
    all_DB = []
    all_DP = []
    all_P = []
    all_rl = []
    all_qm = []
    all_brl = []
    
    bs_range = 1-(1-(np.logspace(1, 0, 100)-1)/9)*.5
    for b_rl in bs_range:
        
        print(country, b_rl)
                
        bs[rl_sub_index] = b_rl
        
        outputs = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_bs)\
        (I0=I0, alphas=alphas, alphas_prime=alphas_prime, betas=betas, A=A, R=R, bs=bs, 
         qm=qm, rl=rl, Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(sample_size))
        
        values = np.array(outputs)
        all_DB += values[:,0].tolist()
        all_DP += values[:,1].tolist()
        all_P += values[:,2].tolist()
        all_rl += values[:,3].tolist()
        all_qm += values[:,4].tolist()
        all_brl += (np.ones(values.shape[0])*b_rl).tolist()
    
    arr1 = np.append(np.array([all_DB, all_rl, all_qm, all_brl]).T, np.array(all_P), axis=1)
    arr2 = np.append(arr1, all_DP, axis=1)
    df_data = pd.DataFrame( arr2, 
                           columns=['DB', 'rl', 'qm', 'brl']+
                           ['indi_P_'+str(indi) for indi in range(len(all_P[0]))]+
                           ['indi_D_'+str(indi) for indi in range(len(all_P[0]))])
    df_data = df_data.round(decimals=8)
    df_data.to_csv(home+'/data/chapter_8/simulation/relative_changes/'+country+'.zip', index=False, compression='zip')
    
    




