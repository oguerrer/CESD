"""Complexity Economics and Sustainable Development
   Chapter 9 source code - Figure 9.7 and Table 9.3
   
   Description: produces data on the validation of aid-impact estimates

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

usdgs = df.seriesCode.unique()

variables = ['sdg6_sanita', 'sdg6_water']

metrics_indi = {}
metrics_indi_null = {}
ab_indi = {}
ab_indi_null = {}
ac_indi = {}
ac_indi_null = {}

output = []
for country in countries:
    
    print(country)
    
    dft = df[df.countryCode==country]
    df_effects = pd.read_csv(home+'/data/chapter_9/simulation/effects_aid/'+country+'.csv')
    df_nulls = pd.read_csv(home+'/data/chapter_9/simulation/effects_synthetic/'+country+'.zip', compression='gzip')
    df_params = pd.read_csv(home+"/data/chapter_9/simulation/parameters/"+country+".csv")
    df_aid = pd.read_csv(home+"/data/chapter_9/simulation/budgets/aid/"+country+".csv")
    
    
    if sum(df_effects.seriesCode.isin(variables)) > 0 and df_aid[df_aid.sdg==6][colYears].values.sum() > 0:
        df_effects['sdg'] = [int(c.split('_')[0][3::]) for c in df_effects.seriesCode]
        df_effects = df_effects[df_effects.seriesCode.isin(variables)]
        df_effects.reset_index(inplace=True)
        df_nulls['sdg'] = [int(c.split('_')[0][3::]) for c in df_nulls.seriesCode]
        df_nulls = df_nulls[df_nulls.seriesCode.isin(variables)]
        df_nulls.reset_index(inplace=True)
        
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
            
        
        metrics_indi[country] = {}
        metrics_indi_null[country] = {}
        ab_indi[country] = {}
        ab_indi_null[country] = {}
        ac_indi[country] = {}
        ac_indi_null[country] = {}
        for index, row in df_effects.iterrows():
                        
            ab_indi[country][row.seriesCode] = area_baseline[index]
            ac_indi[country][row.seriesCode] = area_counter[index]
            null_base = df_nulls[df_nulls.seriesCode==row.seriesCode].area_baseline_null.values
            null_counter = df_nulls[df_nulls.seriesCode==row.seriesCode].area_counter_null.values
            ab_indi_null[country][row.seriesCode] = null_base
            ac_indi_null[country][row.seriesCode] = null_counter
            
            metrics_indi[country][row.seriesCode] = 100*area_counter[index]/area_baseline[index]
            null_base = df_nulls[df_nulls.seriesCode==row.seriesCode].area_baseline_null.values
            null_counter = df_nulls[df_nulls.seriesCode==row.seriesCode].area_counter_null.values
            metrics_indi_null[country][row.seriesCode] = 100*null_counter/null_base




output = []
for country in metrics_indi.keys():
    for variable in metrics_indi[country].keys():
        new_row = [country]
        new_row.append(variable)
        new_row.append(metrics_indi[country][variable])
        new_row.append(np.percentile(metrics_indi_null[country][variable], 90))
        new_row.append(np.percentile(metrics_indi_null[country][variable], 95))
        new_row.append(np.percentile(metrics_indi_null[country][variable], 99))
        output.append(new_row)
         
      
            
dfn = pd.DataFrame(output, columns=['countryCode', 'seriesCode', 'statistic', 'p90', 'p95', 'p99'])
dfn.sort_values(by=['statistic'], inplace=True)
dfn.reset_index(inplace=True)
dfn.to_csv(home+"/data/chapter_9/results/validation.csv", index=False)





