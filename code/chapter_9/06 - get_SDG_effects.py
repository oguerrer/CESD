"""Complexity Economics and Sustainable Development
   Chapter 9 source code - Figure 9.5
   
   Description: produces data on SDG-level aid impacts

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7
"""
import os, sys, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
home =  os.getcwd()[:-14]
sys.path.append(home+'/code/')






df = pd.read_csv(home+"/data/chapter_9/indicators.csv")
colYears = (np.array([year for year in range(2000, 2014)])).astype(str)

all_indis = df.seriesCode.unique()
all_sdgs = sorted(df.sdg.unique())

countries = df.countryCode.unique()
country2group = dict(zip(df.countryCode, df.group))



area_baseline_group_sdgs = dict([(group, dict([(sdg, 0) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])
area_counter_group_sdgs = dict([(group, dict([(sdg, 0) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])
area_baseline_group_sdgs_null = dict([(group, dict([(sdg, np.zeros(1000)) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])
area_counter_group_sdgs_null = dict([(group, dict([(sdg, np.zeros(1000)) for sdg in sorted(df.sdg.unique())])) for group in df.group.unique()])

output = []
for country in countries:
    
    print(country)
    
    dft = df[df.countryCode==country]
    df_effects = pd.read_csv(home+'/data/chapter_9/simulation/effects_aid/'+country+'.csv')
    df_nulls = pd.read_csv(home+'/data/chapter_9/simulation/effects_synthetic/'+country+'.zip', compression='gzip')
    df_params = pd.read_csv(home+"/data/chapter_9/parameters/"+country+".csv")
    df_aid = pd.read_csv(home+"/data/chapter_9/budgets/aid/"+country+".csv")
    df_exp_tot = pd.read_csv(home+"/data/chapter_9/budgets/total/"+country+".csv")
    df_exp = pd.read_csv(home+"/data/chapter_9/budgets/expenditure/"+country+".csv")
    
    aid_filter = df_aid[df_aid[colYears].sum(axis=1) > 0].sdg.values
    df_effects['sdg'] = [int(c.split('_')[0][3::]) for c in df_effects.seriesCode]
    df_effects = df_effects[df_effects.sdg.isin(aid_filter)]
    df_effects.reset_index(inplace=True)
    df_nulls['sdg'] = [int(c.split('_')[0][3::]) for c in df_nulls.seriesCode]
    df_nulls = df_nulls[df_nulls.sdg.isin(aid_filter)]
    df_nulls.reset_index(inplace=True)
    df_exp_tot = df_exp_tot[df_exp_tot.sdg.isin(aid_filter)]
    df_exp_tot.reset_index(inplace=True)
    df_exp = df_exp[df_exp.sdg.isin(aid_filter)]
    df_exp.reset_index(inplace=True)
    
    # Parameters
    alphas = df_params.alpha.values.copy()
    alphas_prime = df_params.alpha_prime.values.copy()
    betas = df_params.beta.values.copy()
    T = int(df_params['T'].values[0])
    num_years = int(df_params.years.values[0])
    
    baselines = [c for c in df_effects.columns if 'baseline' in c]
    counters = [c for c in df_effects.columns if 'counter' in c]
    
    series_baseline = df_effects[baselines].values 
    series_counter = df_effects[counters].values
    area_baseline = np.sum([serie - serie.min() for serie in series_baseline], axis=1)
    area_counter = np.sum([series_baseline[i] - serie_c for i, serie_c in enumerate(series_counter)], axis=1)
    
    effects = area_counter/area_baseline
    statistics = 100*effects
    
    series_baseline = df_nulls[baselines].values 
    series_counter = df_nulls[counters].values
    df_nulls['area_baseline_null'] = np.sum([serie - serie.min() for serie in series_baseline], axis=1)
    df_nulls['area_counter_null'] = np.sum([series_baseline[i] - serie_c for i, serie_c in enumerate(series_counter)], axis=1)
    
    
    for index, row in df_effects.iterrows():
        sdg = row.sdg
        area_baseline_group_sdgs[country2group[country]][sdg] += area_baseline[index]
        area_counter_group_sdgs[country2group[country]][sdg] += area_counter[index]
        
        area_baseline_null = df_nulls[df_nulls.seriesCode==row.seriesCode].area_baseline_null.values
        area_baseline_group_sdgs_null[country2group[country]][sdg] += area_baseline_null
        area_counter_null = df_nulls[df_nulls.seriesCode==row.seriesCode].area_counter_null.values
        area_counter_group_sdgs_null[country2group[country]][sdg] += area_counter_null







new_rows = []
for group in df.group.unique():
    for sdg in all_sdgs:
        if sum(area_baseline_group_sdgs_null[group][sdg]) > 0 and area_baseline_group_sdgs[group][sdg] > 0:
            statistic = 100*area_counter_group_sdgs[group][sdg]/area_baseline_group_sdgs[group][sdg]
            nulls = 100*area_counter_group_sdgs_null[group][sdg]/area_baseline_group_sdgs_null[group][sdg]
            per99 = np.percentile(nulls, 99)
            per95 = np.percentile(nulls, 95)
            new_row = [group, sdg, statistic, per99, per95, nulls.min(), nulls.max()]
            new_rows.append(new_row)

dfn = pd.DataFrame(new_rows, columns=['group', 'sdg', 'statistic', 'per99', 'per95', 'min_nulls', 'max_nulls'])
dfn.to_csv(home+"/data/chapter_9/results/effect_sdgs.csv", index=False)



