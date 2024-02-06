"""Complexity Economics and Sustainable Development
   Chapter 9 source code - Figure 9.4
   
   Description: produces data on country-level aid impacts

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



countries = df.countryCode.unique()


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
    
    effect = area_counter/area_baseline
    statistic = 100*np.mean(effect)
    change_indis = (series_baseline.mean(axis=1) - series_counter.mean(axis=1))/series_counter.mean(axis=1)
    change_bud = df_aid.values[:,1::].sum()/df_exp.values[:,1::].sum()
    elasticity = change_indis.mean() / change_bud
    
    series_baseline = df_nulls[baselines].values 
    series_counter = df_nulls[counters].values
    area_baseline = np.sum([serie - serie.min() for serie in series_baseline], axis=1)
    area_counter = np.sum([series_baseline[i] - serie_c for i, serie_c in enumerate(series_counter)], axis=1)
    df_nulls['effects'] = area_counter/area_baseline
    statistics_null = 100*df_nulls.groupby('simulation').mean().effects.values
    
        
    output.append( [country, statistic, np.mean(statistics_null), np.std(statistics_null), 
                    np.percentile(statistics_null, 90), np.percentile(statistics_null, 95), 
                    np.percentile(statistics_null, 99), np.percentile(statistics_null, 5),
                    np.min(statistics_null), np.max(statistics_null), 
                    df_aid.values[:,1::].sum(), change_bud, elasticity] )



    
dfn = pd.DataFrame(output, columns=['countryCode', 'statistic', 'mean0', 'std0', 'p90', 'p95', 'p99', 'p05', 'min_val', 'max_val', 'aid', 'change_bud', 'elasticity'])
dfn.sort_values(by=['statistic'], inplace=True)
dfn.reset_index(inplace=True)
dfn.to_csv(home+"/data/chapter_9/results/effect_country.csv", index=False)


