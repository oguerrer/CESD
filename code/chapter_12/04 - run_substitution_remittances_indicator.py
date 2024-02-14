"""Complexity Economics and Sustainable Development
   Chapter 12 source code
   
   Description: runs the simulations on the substitutability of remittances

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7

This chapter uses an extended version of the PPI library that allows 
for alternative sources of income that directly affect certain development 
indicators among poor households.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')
import ppip # extended version of PPI





# Dataset
df = pd.read_csv(home+'/data/chapter_12/indicators.csv')
colYears = [col for col in df.columns if str(col).isnumeric()]
df_exp = pd.read_csv(home+"/data/chapter_12/budget.csv")

dfh = pd.read_csv(home+"/data/chapter_12/household_spending.csv")
dfr = pd.read_csv(home+"/data/chapter_12/remittances.csv")
remit_dict = dict(zip(dfr.columns, dfr.values[0]))
spend_dict = dict(zip(dfh.columns[0:-1], dfh[dfh.right=='total'].values[0][0:-1]))
remit_frac = np.mean([remit_dict[year]/spend for year, spend in spend_dict.items()])



num_years = len(colYears)


sub_periods = 4
T = len(colYears)*sub_periods
parallel_processes = 4
sample_size = 100
sub_sample_size = 1000

series = df[colYears].values
N = len(df)

Imax = np.ones(N)
Imin = np.zeros(N)
R = np.ones(N)
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
success_rates = ppip.get_success_rates(series)
mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
mean_drops[np.isnan(mean_drops)] = 0
aa = np.abs(mean_drops/sub_periods)


# Parameters
dfp = pd.read_csv(home+'/data/chapter_12/parameters.csv')
alphas = dfp.alpha.values
alphas_prime = dfp.alpha_prime.values
betas = dfp.beta.values

# Network
A = np.loadtxt(home+"/data/chapter_12/network.csv", delimiter=',')

# Governance
qm = df.cc.values.copy()
rl = df.rl.values.copy()

# Budget 
Bs = df_exp[colYears].values.mean(axis=1)
Bs = np.tile(Bs, (num_years,1)).T
bi_link = pd.read_csv(home+"/data/chapter_12/budget_linkage.csv")
indi2indx = dict(zip(df.seriesCode, range(len(df))))
prog2indx = dict(zip(df_exp.programCode, range(len(df_exp))))
B_dict = dict([indi2indx[indi], []] for indi in df.seriesCode.values)
for index, row in bi_link.iterrows():
    B_dict[indi2indx[row.seriesCode]].append(prog2indx[row.programCode])
Bs = ppip.get_dirsbursement_schedule(Bs, B_dict, T)


remdict = {}
rights = df.right.unique()
for right in rights:
    if right in dfh.right.values:
        remdict[right] = dfh[dfh.right==right].values[0][0:-1].mean()

Rems = []
for index, row in df.iterrows():
    rems = []
    for period in range(num_years):
        for sub_period in range(sub_periods):
            if row.affected==1:
                rems.append( remdict[row.right]/sub_periods )
            else:
                rems.append(0)
    Rems.append(rems)
Rems = np.array(Rems)




# Baseline
print('running baseline scenario...')
sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppip.run_ppi)\
                (I0, alphas, alphas_prime, betas, remitt=Rems, A=A, R=R, qm=qm, rl=rl,
                 Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(1000)))
tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
I_hatb = np.mean(tsI, axis=0)
df_b = pd.DataFrame(I_hatb, columns=range(Bs.shape[1]))




affected = np.where(df.affected==1)[0]
indis = df.seriesCode[affected].values
bud_subs = []

for i, index in enumerate(affected):

    Rems0 = Rems.copy()*(1-remit_frac)
    Bsc = Bs.copy()
    impact = 100
    
    while impact > 1 or impact < 0:
    
        sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(ppip.run_ppi)\
                        (I0, alphas, alphas_prime, betas, remitt=Rems0, A=A, R=R, qm=qm, rl=rl,
                          Imax=Imax, Imin=Imin, Bs=Bsc, B_dict=B_dict) for itera in range(1000)))
        tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
        I_hatc = np.mean(tsI, axis=0)
        df_c = pd.DataFrame(I_hatc, columns=range(Bs.shape[1]))
        
        area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
        area_counter = np.sum(df_b.iloc[index].values - df_c.iloc[index].values)
        impact = 100*area_counter/area_baseline
        
        print(index, impact)
            
        if impact > 1 or impact < 0:
            Bsc[B_dict[index]] *= 1+impact/100
        
    bud_subs.append( (index, indis[i], Bsc[B_dict[index]].sum()) )




dff = pd.DataFrame(np.array(bud_subs)[:,1::], columns=['seriesCode', 'budget'])
dff.to_csv(home+'/data/chapter_12/simulation/substitution/remittances/indicator_level.csv', index=False)






















