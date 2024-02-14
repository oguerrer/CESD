"""Complexity Economics and Sustainable Development
   Chapter 12 source code - Figures 12.5 and 12.6 & Tables 12.2 and 12.3
   
   Description: produces data to measure the impact of income shocks

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7

This chapter uses an extended version of the PPI library that allows 
for alternative sources of income that directly affect certain development 
indicators among poor households.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
home =  os.getcwd()[:-15]
sys.path.append(home+'/code/')
import ppip # extended version of PPI





# Dataset
df = pd.read_csv(home+'/data/chapter_12/indicators.csv')
colYears = [col for col in df.columns if str(col).isnumeric()]
df_exp = pd.read_csv(home+"/data/chapter_12/budget.csv")
dfr = pd.read_csv(home+'/data/chapter_12/budget_rights.csv')

dfh = pd.read_csv(home+"/data/chapter_12/household_spending.csv")
dfr = pd.read_csv(home+"/data/chapter_12/remittances.csv")
remit_dict = dict(zip(dfr.columns, dfr.values[0]))
spend_dict = dict(zip(dfh.columns[0:-1], dfh[dfh.right=='total'].values[0][0:-1]))
remit_frac = np.mean([remit_dict[year]/spend for year, spend in spend_dict.items()])


rights = sorted(df.right.unique())
indi2right = dict(zip(df.seriesCode, df.right))





df_b = pd.read_csv(home+"/data/chapter_12/simulation/removal/baseline.csv")
df_r = pd.read_csv(home+"/data/chapter_12/simulation/removal/remittances.csv")    
df_h = pd.read_csv(home+"/data/chapter_12/simulation/removal/income.csv")  
df_g = pd.read_csv(home+"/data/chapter_12/simulation/removal/expenditure.csv")





direct_r = {}
direct_h = {}
direct_g = {}    
for index, row in df.iterrows():
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_r.iloc[index].values)
    impact = 100*area_counter/area_baseline
    if row.affected==1:
        direct_r[row.seriesCode] = impact
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_h.iloc[index].values)
    impact = 100*area_counter/area_baseline
    if row.affected==1:
        direct_h[row.seriesCode] = impact
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_g.iloc[index].values)
    impact = 100*area_counter/area_baseline
    if row.affected==1:
        direct_g[row.seriesCode] = impact

dff = pd.DataFrame([item for item in direct_r.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/indicator_level/direct_impact_remittances.csv', index=False)

dff = pd.DataFrame([item for item in direct_h.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/indicator_level/direct_impact_income.csv', index=False)

dff = pd.DataFrame([item for item in direct_g.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/indicator_level/direct_impact_expenditure.csv', index=False)




impact_rights_r = {}
impact_rights_h = {}       
impact_rights_g = {}
for right, group in df.groupby('right'):
    index = group.index
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_r.iloc[index].values)
    impact = 100*area_counter/area_baseline
    impact_rights_r[right] = impact
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_h.iloc[index].values)
    impact = 100*area_counter/area_baseline
    impact_rights_h[right] = impact
    area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
    area_counter = np.sum(df_b.iloc[index].values - df_g.iloc[index].values)
    impact = 100*area_counter/area_baseline
    impact_rights_g[right] = impact

dff = pd.DataFrame([item for item in impact_rights_r.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/right_level/direct_impact_remittances.csv', index=False)

dff = pd.DataFrame([item for item in impact_rights_h.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/right_level/direct_impact_income.csv', index=False)

dff = pd.DataFrame([item for item in impact_rights_g.items()], columns=['indicator', 'impact'])
dff.to_csv(home+'/data/chapter_12/results/right_level/direct_impact_expenditure.csv', index=False)









num_years = len(colYears)
sub_periods = 4
T = len(colYears)*sub_periods
df_exp = pd.read_csv(home+"/data/chapter_12/budget.csv")
Bs = df_exp[colYears].values.mean(axis=1)
Bs = np.tile(Bs, (num_years,1)).T
bi_link = pd.read_csv(home+"/data/chapter_12/budget_linkage.csv")
indi2indx = dict(zip(df.seriesCode, range(len(df))))
prog2indx = dict(zip(df_exp.programCode, range(len(df_exp))))
B_dict = dict([indi2indx[indi], []] for indi in df.seriesCode.values)
for index, row in bi_link.iterrows():
    B_dict[indi2indx[row.seriesCode]].append(prog2indx[row.programCode])
Bs = ppip.get_dirsbursement_schedule(Bs, B_dict, T)



# Substitutability to mitigate direct impacts
fracs_r = []
df_sr = pd.read_csv(home+"/data/chapter_12/simulation/substitution/remittances/indicator_level.csv")
for index, row in df_sr.iterrows():
    i = np.where(df.seriesCode==row.seriesCode)[0][0]
    total_expenditure = Bs[B_dict[i]].sum()
    frac = 100*row.budget/total_expenditure
    fracs_r.append((row.seriesCode, frac))

dff = pd.DataFrame([item for item in fracs_r], columns=['indicator', 'percentage'])
dff.to_csv(home+'/data/chapter_12/results/indicator_level/direct_substitutability_remittances.csv', index=False)



fracs_i = []
df_si = pd.read_csv(home+"/data/chapter_12/simulation/substitution/income/indicator_level.csv")
for index, row in df_si.iterrows():
    i = np.where(df.seriesCode==row.seriesCode)[0][0]
    total_expenditure = Bs[B_dict[i]].sum()
    frac = 100*row.budget/total_expenditure
    fracs_i.append((row.seriesCode, frac))

dff = pd.DataFrame([item for item in fracs_i], columns=['indicator', 'percentage'])
dff.to_csv(home+'/data/chapter_12/results/indicator_level/direct_substitutability_income.csv', index=False)








dfp = pd.read_csv(home+"/data/chapter_12/social_spending.csv")
progs_desc = dict(zip(dfp['program code'], dfp['Desc 3']))
progs_rights = dict([(code, set(group['right'].values)) for code, group in dfp.groupby('program code')])
progs_indis = dict([(code, group[df.seriesCode.values].values.max(axis=0).sum()) for code, group in dfp.groupby('program code')])



progs_rights = dict([(code, set(group['right'].values)) for code, group in dfp.groupby('program code')])
program2idx = dict(zip(df_exp.programCode, df_exp.index))
rights_progs = dict([(r,[]) for r in rights])
for program, rs in progs_rights.items():
    for right in list(rs):
        if program in program2idx:
            rights_progs[right.lower()].append(program2idx[program])





# Substitutability to mitigate system-wide impacts
fracs_r = []
df_sr = pd.read_csv(home+"/data/chapter_12/simulation/substitution/remittances/right_level.csv")
for index, row in df_sr.iterrows():
    i = np.where(df.right==row.right)[0][0]
    total_expenditure = Bs[rights_progs[row.right]].sum()
    frac = 100*row.budget/total_expenditure
    fracs_r.append((row.right, frac))

dff = pd.DataFrame([item for item in fracs_r], columns=['right', 'percentage'])
dff.to_csv(home+'/data/chapter_12/results/right_level/direct_substitutability_remittances.csv', index=False)


fracs_i = []
df_si = pd.read_csv(home+"/data/chapter_12/simulation/substitution/income/right_level.csv")
for index, row in df_si.iterrows():
    i = np.where(df.right==row.right)[0][0]
    total_expenditure = Bs[rights_progs[row.right]].sum()
    frac = 100*row.budget/total_expenditure
    fracs_i.append((row.right, frac))

dff = pd.DataFrame([item for item in fracs_i], columns=['right', 'percentage'])
dff.to_csv(home+'/data/chapter_12/results/right_level/direct_substitutability_income.csv', index=False)












dfs = pd.read_csv(home+"/data/chapter_12/budget.csv")
programs = dfs.programCode.values
dfl = pd.read_csv(home+'/data/chapter_12/budget_linkage.csv')
files = os.listdir(home+"/data/chapter_12/simulation/reduction/")
impact_programs_direct = []
impact_programs_system = []
for file in files:
    if 'program' in file:
        i = int(file.split('.')[0].split('_')[-1])
        df_p = pd.read_csv(home+"/data/chapter_12/simulation/reduction/"+file)
        indis = dfl[dfl.programCode==programs[i]].seriesCode.values
        index = df.seriesCode.isin(indis).values
        area_baseline = np.sum(df_b.iloc[index].values - df_b.iloc[index].values.min())
        area_counter = np.sum(df_b.iloc[index].values - df_p.iloc[index].values)
        impact = 100*area_counter/area_baseline
        impact_programs_direct.append((impact, programs[i]))
        area_baseline = np.sum(df_b.values - df_b.values.min())
        area_counter = np.sum(df_b.values - df_p.values)
        impact = 100*area_counter/area_baseline
        impact_programs_system.append((impact, programs[i]))


dff = pd.DataFrame([item for item in impact_programs_direct], columns=['impact', 'programId'])
dff.to_csv(home+'/data/chapter_12/results/program_level/direct_impact_reduction.csv', index=False)

dff = pd.DataFrame([item for item in impact_programs_system], columns=['impact', 'programId'])
dff.to_csv(home+'/data/chapter_12/results/program_level/systemic_impact_reduction.csv', index=False)






