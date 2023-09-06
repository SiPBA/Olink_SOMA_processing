# -*- coding: utf-8 -*-
"""
Script de generación de resultados finales a partir de los resultados obtenidos
de los bases de datos OLINK y SOMA.
"""
import pandas as pd
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt

# define the regular expression pattern
pattern = r'.*resultados.*\.csv'

# dataset = 'Olink'
dataset = 'Soma'


metrics = ['Accuracy', 'Especificidad', 'Sensibilidad'] #, 'F1' 'Especificidad', 

# Encontrar los CSV en el directorio y subdirectorios
file_list = [file for file in glob.glob('Resultados 2/' + dataset + '/**/*.csv', recursive=True) if re.match(pattern, file)]
# Convierte cada CSV en una lista de dataframes
data_frames = [pd.read_csv(file, sep=';') for file in file_list]


plt.close('all')

# Concatena todos los dfs en un solo dataframe
df = pd.concat(data_frames, axis=0)
# print(df)

# Regex, solo queremos los CSV con resultados
pattern = r'.*resultados.*\.csv'    
    

df_prov = pd.DataFrame()
results = pd.DataFrame()
setup = []

for i in range(0, len(data_frames)): # 
    df_act = data_frames[i]
    setup.append(df_act.loc[0, 'setup'])
    
    if i==0:
        df_prov = df_act
    else:
        df_prov = pd.concat([df_prov, df_act])



for m, metric in enumerate(metrics):
        
    df_exp = df_prov[df_prov["Metric"] == metric]
    
    df_exp = df_exp.groupby(['selection']).apply(lambda x: x)
    df_exp = df_exp.drop(columns=['Unnamed: 0', 'Metric'])
    melt = pd.melt(df_exp, id_vars=['selection', 'setup'], value_vars=['Ensemble', 'setup']) # baseline
        
    
    results = melt.dropna().reset_index(drop=True)
    
    
    
    results.loc[:, 'setup'] = results.loc[:, 'setup'].replace(' vs ', '-', regex=True)
    # results.loc[:, 'setup'] = results.loc[:, 'setup'].replace('RESTO', 'REST', regex=True)
    
    plt.rcParams.update({'font.size': 6.5})
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax = sns.swarmplot(data=results, x='setup', y='value', hue="selection") 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6])
    ax.set(ylim=(0, 1))
    ax.set_ylabel(metric)
    ax.set_xlabel('Setup')
    ax.figure.savefig('Resultados 2/' + dataset + '/' +
                      metric + ' ft_selection'  + '.png', dpi=200)
    
    
#     fig2, ax2 = plt.subplots(figsize=(20, 6))
#     ax2 = sns.swarmplot(data=results, x='setup', y='value', hue="variable") 
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(handles2[:6], labels2[:6])
#     ax2.set(ylim=(0, 1))
#     ax2.set_ylabel(metric)
#     ax2.set_xlabel('Setup')
#     ax2.figure.savefig('Resultados 2/' + dataset + '/' +
#                       metric + ' classifiers' + '.png')






# ax2 = sns.catplot(data=results, x='setup', y='value', hue="selection", kind='swarm')
# handles2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(handles2[:6], labels2[:6])
# ax2.set(ylim=(0, 1))
# ax2.set_ylabel('Balanced Accuracy')
# ax2.set_xlabel('Setup')





