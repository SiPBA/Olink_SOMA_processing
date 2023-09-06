# -*- coding: utf-8 -*-
"""
Script de análisis y experimentaciones de las bases de datos OLINK y SOMA.
"""

# from sklearnex import patch_sklearn
# patch_sklearn()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt          
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from mrmr import mrmr_classif
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

import func

# =============================================================================
# Configuración
# =============================================================================
np.random.seed(666)

dataset = 'Olink'
# dataset = 'Soma'


# HC
seleccion_proteinas = ['NEFL', 'TPM3', 'ACHE', 'WFDC1', 'FGF20', 'DDC', 'AGRP', 'LATS1', 'PSMG3', 'CHI3L1', 'FURIN', 'WASL', 'Clase']
# seleccion_proteinas = ['NFL', 'IGF-I', 'CO9A3', 'NFH', 'RELN', 'GAPDH, liver', 'HCE004359', 'Hemopexin.1', 'Albumin', 'KCNF1', 'SPB13', 'I5P1', 'BRM1L', 'Fc_MOUSE.44', 'Clase']
experiment = 'HC'

# MSA
# seleccion_proteinas = ['TPM3', 'NEFL', 'PGLYRP1', 'LILRA6', 'CALB2', 'TRDMT1', 'PALM3', 'PSIP1', 'LMNB2', 'TLR4', 'ACHE', 'Clase']
# seleccion_proteinas = ['NFL', 'GAPDH, liver', 'IGF-I', 'EDIL3', 'GAA', 'IGFALS', 'Thrombin', 'PKHB1', 'NC2B', 'Angiostatin', 'Integrin a1b1', 'Adiponectin', 'HPLN1', 'Clase']
# experiment = 'MSA'

# PD
# seleccion_proteinas = ['NEFL', 'SOST', 'TPM3', 'GSTA3', 'LXN', 'CXCL17', 'GRP', 'AGRP', 'CEMIP2', 'RASGRF1', 'CGB3_CGB5_CGB8','Clase']
# seleccion_proteinas = ['NFL', 'GAA', 'Prolylcarboxypeptidase', 'EDIL3', 'NEUR1', 'GAPDH, liver', 'resistin', 'ITIH3', 'Clase']
# experiment = 'PD'

# PSP
# seleccion_proteinas = ['GSTA3', 'CD8A', 'SOWAHA', 'CLIC5', 'PXN', 'Clase']
# seleccion_proteinas = ['PCDA7', 'Integrin aVb8', 'MAST4', 'galactosidase, alpha', 'VPS29', 'Clase']
# experiment = 'PSP'

# Otras 
# seleccion_proteinas = ['NEFL', 'TPM3', 'SOST', 'GLP1R', 'LXN', 'AK2', 'WFDC1', 'TP53INP1', 'GRHPR', 'FGF20', 'CALB2', 'DDC', 'Clase']
# seleccion_proteinas = ['NFL', 'NFH', 'GAA', 'RAB4B', 'PLCD3', 'S100A13', 'Albumin', 'Apo C-I', 'Fc_MOUSE.44', 'HCE004359', 'GAPDH, liver', 'Clase']
# experiment = 'Otras'


# HC+PSP vs MSA+PD
# seleccion_proteinas = ['GLP1R', 'AK2', 'TP53INP1', 'GRHPR','Clase']
# experiment = 'HC+PSP vs MSA+PD'

# TODOS
# seleccion_proteinas = ['NEFL', 'TPM3', 'ACHE', 'WFDC1', 'FGF20', 'PI3', 'CALB2', 'CRH', 'DDC', 'AGRP', 'GRP', 'PSMG3', 'LATS1', 'AMOTL2', 'GABARAP', 'MTHFSD', 'ARL2BP', 'FURIN', 'CHI3L1', 'MFGE8', 'GSTA3', 'PODXL2', 'WASL', 'JCHAIN', 'SIGLEC7', 'PSIP1', \
#                         'PGLYRP1', 'LILRA6', 'SOST', 'FNDC1', 'SLURP1', 'ITGA6', 'MMP7', 'MAPK9', 'TRDMT1', 'PALM3', 'LMNB2', 'TLR4', 'ADIPOQ', 'CD8A', 'RICTOR', 'SPINT3', 'HDAC8', \
#                         'CXCL9', 'CCL11', 'LXN', 'HCLS1', 'MMP12', 'CXCL17', 'CEMIP2', 'PDP1', 'NECAP2', 'LAMTOR5', 'FOSB', 'ELN', 'IL24', 'IL20RB', 'CGB3_CGB5_CGB8', \
#                         'SOWAHA', 'CLIC5', 'SELPLG', 'DPT', 'PPL', 'GLP1R', 'GAST', 'Clase']
                       

                       
# seleccion_proteinas = ['NFL', 'IGF-I', 'CO9A3', 'NFH', 'RELN', 'GAPDH, liver', 'HCE004359', 'Hemopexin.1', 'Albumin', 'KCNF1', 'SPB13', 'I5P1', 'BRM1L', 'Fc_MOUSE.44', \
#                         'IGF-I', 'EDIL3', 'GAA', 'IGFALS', 'Thrombin', 'PKHB1', 'NC2B', 'Angiostatin', 'Integrin a1b1', 'Adiponectin', 'HPLN1', \
#                         'Prolylcarboxypeptidase', 'NEUR1', 'resistin', 'ITIH3', \
#                         'PCDA7', 'Integrin aVb8', 'MAST4', 'galactosidase, alpha', 'VPS29', \
#                         'RAB4B', 'PLCD3', 'S100A13', 'Apo C-I','Clase']
# experiment = 'Todos'





# experiment = 'HC+PD vs MSA+PSP'

# -- Selección de biomarcadores
# HC vs MSA
# seleccion_proteinas = ['NEFL', 'TPM3', 'ACHE', 'WFDC1', 'FGF20', 'Clase'] # , 'PI3', 'CALB2', 'CRH'
# seleccion_proteinas = ['NFL', 'RELN', 'CO9A3', 'NFH', 'SCG1', 'IGF-I', 'LIGO3', 'Integrin a1b1', 'WFDC1', 'Clase']

# HC vs PD
# seleccion_proteinas = ['FGF20', 'DDC', 'AGRP', 'LATS1', 'PSMG3','Clase'] # 'AGRP', 'GRP', 'LATS1', 'MTHFSD', 'PSMG3', 'GABARAP', 'ARL2BP'
# seleccion_proteinas = ['GAPDH, liver', 'HCE004359', 'S100A13', 'FUT9', 'ATG7', 'LGMN.1', 'Dynorphin A (1-17)', 'C3b', 'SGTB', 'C4', 'ABHD4', 'Clase'] 

# HC vs PSP
# seleccion_proteinas = ['NEFL', 'TPM3', 'DDC', 'CHI3L1', 'FURIN', 'Clase'] # 'CHI3L1', 'DDC', 'GSTA3', 'PODXL2'
# seleccion_proteinas = ['NFL', 'NFH', 'NEUR1', 'KCNF1', 'KCE1L:N-term', 'M4K1', 'PLOD2', 'SPB13', 'Antithrombin III', 'LIGO3', 'REM1', 'I5P1', 'Clase'] # , 'PLOD2'

# HC vs RESTO
# seleccion_proteinas = ['NEFL', 'CHI3L1', 'TPM3', 'TPSAB1', 'FGF20', 'Clase'] # 'CHI3L1', 'DDC', 'TPSAB1', 'BRD2', 'SIGLEC7', 'DNM3', 'FGF20',
# seleccion_proteinas = ['RELN', 'LIGO3', 'SCG1', 'Albumin', 'GGH', 'NFH', 'CO9A3', 'HCE004359', 'Periostin', 'NFL', 'FETUB', 'Apo A-II', 'WFDC1', 'Fc_MOUSE.44', 'Ephrin-A5', 'Clase']




# Path donde se guardan los resultados
path_resultados = 'Resultados 2/' + dataset +'/'  


# -- Lectura y preprocesamiento de los datasets

if dataset == 'Olink':
    olink = pd.read_csv('olink.csv')
    olink = olink.drop(columns="Unnamed: 0")
    olink = olink[olink["SampleID"].str.contains("PD - 5") == False]
    prot = olink.drop(columns=['SampleID'])
    df = olink
    
if dataset == 'Soma':
    soma = pd.read_csv('soma.csv')
    soma = soma.drop(columns=['Unnamed: 0'])
    df = soma
    soma = soma[soma["SampleID"].str.contains("PD - 5") == False]
    soma = soma.replace('. ','.', regex=True)
    df = df.replace('. ','.', regex=True)
    
    
    prot = soma.drop(columns=['SampleID'])
    
    # Convert columns to numeric except 'column_to_exclude'
    columns_to_exclude = ['Class']
    
    for col in prot.columns:
        if col not in columns_to_exclude:
            prot[col] = pd.to_numeric(prot[col], errors='coerce')




prot['Clase'] = prot['Class']
prot = prot.drop(columns=['Class'])






# -- Discretización de la variable Class

data = prot.loc[:, 'Clase']
values = np.array(data)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
le_name_mapping = dict(zip(label_encoder.classes_, 
                            label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

prot['Clase'] = integer_encoded

nombres = prot.columns



X = prot.drop(columns=['Clase']) # , 'SampleID'
y = prot.loc[:, 'Clase']
classes = y

subjects = df.loc[:, 'SampleID']
nombres = prot.columns


selection = prot.loc[:, seleccion_proteinas]




# =============================================================================
# PANDAS PROFILING
# =============================================================================
# profile = ProfileReport(selection, title="Profiling Report", minimal=False)
# profile.to_file(path_resultados + experiment + ' report' + '.html')

# =============================================================================
# PAIRPLOT
# =============================================================================
# pair plot
selection.loc[:, 'Clase'] = label_encoder.inverse_transform(selection.loc[:, 'Clase'])
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(18, 18)})
# g = sns.pairplot(selection, hue="Clase")
# g.map_lower(sns.kdeplot, levels=4, color=".2")
# g.fig.set_size_inches(28, 28)
# plt.savefig(path_resultados + experiment + '/Análisis/' +
#             experiment + ' pairplot selection' + '.png', dpi=199)
# g = sns.PairGrid(selection, vars=df.columns[:-1])
# g.map(sns.stripplot, jitter=True, size=3)

# plt.show()



# =============================================================================
# BOXPLOT COMPLETO
# =============================================================================
sns.set(rc={'figure.figsize':(12, 10)})
sns.set_style("whitegrid")
plt.figure()
selection_long = selection.melt(id_vars=['Clase'])

if dataset=='Olink':
    sns.boxplot(data=selection_long, y="value", x="variable", orient="v", hue="Clase") 
if dataset=='Soma':
    sns.boxplot(data=selection_long, y="value", x="variable", orient="v", hue="Clase", showfliers=False) 
plt.xlabel('Proteínas')
plt.ylabel('Valores')
plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' boxplot completo' + '.png')

# =============================================================================
# PCA DE PROTEÍNAS
# =============================================================================
plt.figure()
sns.set(rc={'figure.figsize':(10, 10)})
sns.set_style("whitegrid")
# Show first two principal components with scaler
pca = PCA()
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
# plt.figure(figsize=(8,6))
Xt = pipe.fit_transform(selection.drop(columns=['Clase']).T)
# plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)
# plt.legend(handles=plot.legend_elements()[0], labels=list(label_encoder.inverse_transform(np.unique(y))))
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("First two principal components after scaling")
# plt.show()

labels = selection.columns[:-1].tolist()
plt.figure(figsize=(10, 8))

scatter=plt.scatter(
    Xt[:, 0], Xt[:, 1], 
    marker='o', 
    s=8
    )

plt.xlabel("PC1", fontsize=10)
plt.ylabel("PC2", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)



for label, x, y in zip(labels, Xt[:, 0], Xt[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(10, 5),
        textcoords='offset points', ha='right', va='bottom', size = 7
        )
plt.legend('', frameon=False)# , loc='upper left'
plt.show()
plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' PCA prot' + '.png')

# =============================================================================
# BOXPLOT DE CADA PROT SELECCIÓN
# =============================================================================


sns.set(rc={'figure.figsize':(5, 5)})
sns.set_style("whitegrid")
for i, pr in enumerate(selection.columns[:-1]):
    plt.figure()
    df_subselection = selection.loc[:, [pr, 'Clase']]
    df_subselection_long = df_subselection.melt(id_vars=['Clase'])
    # sns.boxplot(selection, x=pr, y="Clase")
    if dataset=='Olink':
        sns.boxplot(data=df_subselection_long, y="value", x="variable", orient="v", hue="Clase")
    if dataset=='Soma':
        sns.boxplot(data=df_subselection_long, y="value", x="variable", orient="v", hue="Clase", showfliers=False)
    plt.xlabel('')
    plt.ylabel('Valor')
    plt.savefig(path_resultados + experiment + '/Análisis/' +
                experiment + ' boxplot selection ' + pr + '.png')


# =============================================================================
# CLUSTERING
# =============================================================================

print('corr')
X_selection = selection.drop(columns=['Clase'])
plt.figure(figsize=(15, 10))
correlations = X_selection.corr()
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);

# plt.savefig(path_resultados + 
#             ' corr' +
#             '.png')
plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' corr' + '.png')




print('cluster')
# plt.figure(figsize=(15, 8))
plt.figure(figsize=(8, 13))
dissimilarity = 1 - abs(correlations)
Z = linkage(squareform(dissimilarity), 'complete')

dendrogram(Z, labels=X_selection.columns, orientation='right'); # ,leaf_rotation=90

# plt.savefig(path_resultados + 
#             'Dendro' + '.png')
# plt.savefig(path_resultados + 
#             'Dendro' + '.pdf')
plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' clustering' + '.png')


# Clusterize the data
threshold = 0.9
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
labels

# Keep the indices to sort labels
labels_order = np.argsort(labels)

# Build a new dataframe with the sorted columns
for idx, i in enumerate(X_selection.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(X_selection[i])
    else:
        df_to_append = pd.DataFrame(X_selection[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)
        
print('clustered corr')    
plt.figure(figsize=(25, 20))
correlations = clustered.corr()
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);        
# plt.savefig(path_resultados + 
#             'Clustered corr' + '.png')
# plt.savefig(path_resultados + 
#             'Clustered corr' + '.pdf')
plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' corr2' + '.png')


plt.figure(figsize=(25, 20)) # 6, 4
sns.clustermap(correlations, method="complete", cmap='RdBu', annot=True, 
                annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(25, 20))

plt.savefig(path_resultados + experiment + '/Análisis/' +
            experiment + ' dendrocorr' + '.png')

from sklearn.metrics import silhouette_score
# silhouette_avg = silhouette_score(clustered, labels)
# print("Clustered Silhouette Score:", silhouette_avg)
# from sklearn.metrics import davies_bouldin_score
# davies_bouldin_index = davies_bouldin_score(clustered, labels)
# print("Clustered Davies-Bouldin Index:", davies_bouldin_index)

silhouette_avg = silhouette_score(correlations, labels)
print("Correlation Silhouette Score:", silhouette_avg)
from sklearn.metrics import davies_bouldin_score
davies_bouldin_index = davies_bouldin_score(correlations, labels)
print("Correlation Davies-Bouldin Index:", davies_bouldin_index)

from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(correlations, labels)
print("Calinski-Harabasz Score:", ch_score)


# El índice Davies Boulding mide la separabilidad media de una agrupación en contraposición frente a la de la agrupación más cercana, y es mejor cuanto más pequeño sea el resultado:
# La medida de silueta cuantifica la distancia de una agrupación a las muestras de las demás agrupaciones, mejor cuando el valor sea cercano a 1:
    
# =============================================================================
# CLF SVM LIN HUELLA
# =============================================================================

# X = selection.drop(columns=['Clase'])
# y = selection.loc[:, 'Clase']
# y = label_encoder.fit_transform(y)

# X = np.asarray(X, dtype=float) 
# y = np.array(y)

# n_fts = len(selection.columns.tolist())
# # Validación cruzada 
# from sklearn.model_selection import RepeatedStratifiedKFold
# splits = 5
# reps = 5
# split = RepeatedStratifiedKFold(n_splits = splits, 
#                                     n_repeats = reps, 
#                                     random_state = None) 

# # Variables de salida 

# Caracteristicas_cv = np.nan * np.ones([0, n_fts]) # Matriz 
# Contador_Iteracion = 0

# mean_fpr = np.linspace(0, 1, 100)
# tprs = []
# aucs = []


# acc = pd.DataFrame()
# aucdf = pd.DataFrame()
# resultados = pd.DataFrame()
# resultados_finales = pd.DataFrame()
# caracteristicas_finales = pd.DataFrame()
# weights_todos = pd.DataFrame()
# tpr_df = pd.DataFrame()
# fpr_df = pd.DataFrame()

# for i, ft in enumerate(selection.columns[:-1]):
    
#     iteracion_clf = 0
    
#     confmat_aux = np.array([0, 0, 0, 0])
#     for train, test in split.split(X, y):
        
        
#         Contador_Iteracion = Contador_Iteracion + 1
#         porcentaje = Contador_Iteracion/(splits * 
#                                           reps *
#                                           len(selection.columns[:-1])) * 100
#         print('')
#         print('Completado: ' + str(Contador_Iteracion),
#               '/', 
#               splits * reps * len(selection.columns[:-1]), 
#               '(',
#               round(porcentaje, 2),
#               '% )') # Lleva la cuenta de la simulación
        

#         X_train = X[train, :]
#         X_test = X[test, :]
#         y_train = y[train]
#         y_test = y[test]
#         X_trainFS = X_train[:, i].reshape(-1, 1)
#         X_testFS = X_test[:, i].reshape(-1, 1)

#         print(ft)
    
    
#         # Clasificación 
#         clf, y_pred, Scores, Params, weights = func.clf_lin(X_trainFS, 
#                                                     y_train, 
#                                                     X_testFS) 
    
#         weights_todos = weights_todos.append(pd.DataFrame(weights), ignore_index=True)
        
     
#         from sklearn.metrics import  classification_report
#         report = classification_report(y_test, y_pred)
#         print(report)
        
        

    
#         # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, drop_intermediate=False) #♣ , pos_label=2
#         # AUC = metrics.auc(fpr, tpr)
        
        

#         tmp_confmat_aux = metrics.confusion_matrix(y_test,
#                                                     y_pred,
#                                                     labels=np.unique(y)).flatten()
    
    
#         if 'confmat_aux' not in locals():
#             confmat_aux = tmp_confmat_aux
#         else:
#             confmat_aux = np.vstack((confmat_aux, tmp_confmat_aux))
    
#         tn, fp, fn, tp = tmp_confmat_aux 
#         # cm = metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y)).ravel()
    
#         print('')
#         print('Iteración clasificación: ', iteracion_clf)
#         acc.loc[iteracion_clf, ft] = round(metrics.accuracy_score(y_test, y_pred), 3)
#         # aucdf.loc[iteracion_clf, ft] = round(AUC)        
#         print('')
#         print('### --------------------------------------- ###')
#         print('')
#         iteracion_clf += 1
    
    
#     # tpr_df = pd.concat([tpr_df, pd.DataFrame(tpr)], axis=1)
#     # fpr_df = pd.concat([fpr_df, pd.DataFrame(fpr)], axis=1)
    
    
#     # =============================================================================
#     #     Resultados
#     # =============================================================================
    
    
#     resultados_selection = pd.DataFrame()
#     resultados_selection = resultados_selection.append(acc.mean(axis=0), 
#                                                         ignore_index=True)
#     resultados_selection = resultados_selection.append(aucdf.mean(axis=0), 
#                                                         ignore_index=True)
#     resultados_selection.loc[:, 'Metric'] = ['Accuracy', 'AUC']
            
#     resultados_selection = resultados_selection.round(3) 
        
    
    
 
    
  
#     resultados_selection.loc[:, 'setup'] = experiment
#     resultados_selection.loc[:, 'ft'] = ft
    
#     resultados_selection.to_csv(path_resultados + 
#                                 experiment +
#                                 ' ' +
#                                 ' resultados ' + 
#                                 '.csv',
#                                 sep = ";")
    
#     resultados_finales = resultados_finales.append(resultados_selection, 
#                                                     ignore_index=True)
    

# # plt.figure() # figsize=(9, 9)
# sns.set_style("white")
# # for i in range(len(fpr_df.columns)):
   
# #     plt.plot(fpr_df.iloc[:, i], tpr_df.iloc[:, i], label=selection.columns[i] + '. AUC= ' + str(resultados_selection.iloc[1, i])) # ,label="Logistic Regression, AUC="+str(auc)
    
    
# # plt.plot([0, 1], [0, 1], 'k--', label='Clf. aleatorio')
# # plt.legend()
# # plt.savefig(path_resultados + experiment + '/Análisis/' +
# #             experiment + ' ROC prot' + '.png')
#   # HACER BOXPLOT DE RESULTADOS DE CLF: ACC, AUC? PROBAR ROC

# resultados_selection = resultados_selection[resultados_selection["Metric"].str.contains("AUC") == False]
# resultados_selection = resultados_selection.drop(columns=['Metric', 'setup', 'ft'])
# resultados_selection_long = resultados_selection.melt()
# plt.figure()
# sns.color_palette("rocket")
# sns.barplot(data=resultados_selection_long, x="variable", y="value", palette='rocket')
# plt.savefig(path_resultados + experiment + '/Análisis/' +
#             experiment + ' barplot accuracy' + '.png')





# # =============================================================================
# # 
# # =============================================================================
# # Establecer el estilo de seaborn
# sns.set(style='ticks')

# # Obtener la paleta Tab10 de seaborn
# colors = sns.color_palette('tab10')

# # Crear el violin plot con colores personalizados
# sns.violinplot(data=prot.values, palette=colors, inner='quartile')

# # Ajustar el tamaño de la figura
# plt.figure(figsize=(8, 6))

# # Añadir título y etiquetas de los ejes
# plt.xlabel('Valores')
# plt.ylabel('Nivel de cuantificación')

# # Personalizar el estilo de los componentes del gráfico
# sns.despine(trim=True, left=True)
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # Ajustar los márgenes del gráfico
# plt.tight_layout()
# plt.xticks([])

# # Mostrar el violin plot
# # plt.show()


# # =============================================================================
# # 
# # =============================================================================


# # Configurar el tamaño del gráfico
# plt.figure(figsize=(16, 8))
# df = prot

# # Crear el swarm plot
# class_colors = {'A': 'red', 'B': 'green', 'C': 'blue', 'D': 'yellow'}

# sns.swarmplot(data=prot, 
#               size=2,
#               # palette=sns.color_palette("tab10"),
#               edgecolor='black',
#               )

# # Configurar las etiquetas y los límites del eje x
# num_variables = nombres.size
# plt.xlabel('Proteínas')
# # plt.xlim(0, num_variables)
# plt.xticks([])
# plt.ylabel('Nivel de cuantificación')

# # Ajustar las etiquetas del eje x para evitar la superposición
# # plt.xticks(np.arange(0, 2929, step=100), rotation=90)

# # Mostrar el gráfico
# plt.show()





