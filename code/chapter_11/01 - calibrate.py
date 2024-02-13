"""Complexity Economics and Sustainable Development
   Chapter 11 source code
   
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
df = pd.read_csv(home+'data/modeling/granular/simulated_data_integrated_sample.csv')
df_exp = pd.read_csv(home+"data/modeling/granular/budget_integrated.csv")
df_exp = df_exp[df_exp.target.isin(df.target.values.tolist()+df.target2.values.tolist())]
all_targets = sorted(list(set(df.target[~df.target.isnull()].values.tolist() + 
                              df.target2[~df.target2.isnull()].values.tolist())), 
                     key=lambda x: (int(x.split('.')[0]), len(x.split('.')[1]) ,x.split('.')[1]))
targets2ind = dict([(target, i) for i, target in enumerate(all_targets)])
ind2target = dict([(i, target) for i, target in enumerate(all_targets)])
colYears = [col for col in df_exp.columns if col.isnumeric()]




num_years = len(colYears)

sub_periods = 4
T = len(colYears)*sub_periods



series = df[colYears].values
N = len(df)

Imax = None
Imin=None
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
success_rates = ppi.get_success_rates(series)
mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
mean_drops[np.isnan(mean_drops)] = 0
aa = np.abs(mean_drops/sub_periods)

# Network
A = np.loadtxt(home+"data/modeling/granular/network.csv", delimiter=',')

# Governance
qm = df.cc.values.copy()
rl = df.rl.values.copy()

# Budget 
Bs = df_exp[colYears].values.mean(axis=1)
Bs = np.tile(Bs, (num_years,1)).T
B_dict = {}
for index, row in df.iterrows():
    B_dict[index] = []
    for target in df_exp.target.values:
        if row.target == target or row.target2 == target:
            B_dict[index].append(targets2ind[target])
Bs = ppi.get_dirsbursement_schedule(Bs, B_dict, T)


outputs = ppi.calibrate(I0, IF, success_rates, A=A, R=R, qm=qm, rl=rl,  Bs=Bs, B_dict=B_dict, 
          threshold=.9, parallel_processes=60, verbose=False, low_precision_counts=101, increment=1000)

dfc = pd.DataFrame(outputs[1::,:].astype(float), columns=outputs[0])
dfc['years'] = np.ones(N)*np.nan
dfc.loc[0, 'years'] = num_years

# dfc.to_csv(home+'data/modeling/granular/parameters.csv', index=False)





