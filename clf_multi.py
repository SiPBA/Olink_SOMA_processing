# -*- coding: utf-8 -*-
"""
Script de clasificación de las bases de datos OLINK y SOMA multiclase.
"""

from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
import time

import func

#%% MAIN %%#
#%%

# =============================================================================
# Configuración
# =============================================================================

np.random.seed(666)
n_fts = 5 # Características a seleccionar
splits = 5 # Splits de validación cruzada
reps = 5 # Repeticiones de validación cruzada

fts_selection = ['anova', 'mrmr', 'wilcoxon', 'rf']

# dataset = 'Olink'
dataset = 'Soma'

# =============================================================================
# # Lectura de datos 
# =============================================================================
path_resultados = 'Resultados 2/' + dataset +'/' 

if dataset == 'Olink':
    olink = pd.read_csv('olink.csv')
    olink = olink[olink["SampleID"].str.contains("PD - 5") == False]
    olink = olink.drop(columns=['SampleID', 'Unnamed: 0'])
    
if dataset == 'Soma':
    olink = pd.read_csv('soma.csv')
    olink = olink[olink["SampleID"].str.contains("PD - 5") == False]
    olink = olink.drop(columns=['Unnamed: 0', 'SampleID'])
    olink = olink.replace('. ','.', regex=True)

olink['Clase'] = olink['Class']
olink = olink.drop(columns=['Class'])

experimento = 'Multiclase'
path_resultados += experimento + '/'



# Discretización de la variable Class

data = olink.loc[:, 'Clase']
values = np.array(data)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
le_name_mapping = dict(zip(label_encoder.classes_, 
                           label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

olink['Clase'] = integer_encoded
nombres = olink.columns



# Conjuntos y etiquetas 

X = olink.drop(columns=['Clase'])
y = olink.loc[:, 'Clase']

X = np.asarray(X, dtype=float) 
y = np.array(y)




# Validación cruzada 

split = RepeatedStratifiedKFold(n_splits = splits, 
                                   n_repeats = reps, 
                                   random_state = None) 

# Variables

Parametros = [] # Parámetros de clasificación
Caracteristicas_cv = np.nan * np.ones([0, n_fts]) # Matriz 
Contador_Iteracion = 0

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

acc = pd.DataFrame()
sens = pd.DataFrame()
bal = pd.DataFrame()
prec = pd.DataFrame()
f1df = pd.DataFrame()
resultados = pd.DataFrame()
resultados_finales = pd.DataFrame()
caracteristicas_finales = pd.DataFrame()

start_time = time.time()

# %% Validación cruzada

for i, feature_selection in enumerate(fts_selection):

    iteracion_clf = 0
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6)) # Para las curvas roc
    
    confmat_aux = np.array([np.zeros(16)])
    for train, test in split.split(X, y):
        
        
        Contador_Iteracion = Contador_Iteracion + 1
        porcentaje = Contador_Iteracion/(splits * reps  * len(fts_selection)) * 100
        print('Selección: ' + feature_selection)
        print('')
        print('Completado: ' + str(Contador_Iteracion),
              '/', 
              splits * reps * len(fts_selection), 
              '(',
              round(porcentaje, 2),
              ' %)') # Lleva la cuenta de la simulación
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Elapsed {:02d}:{:02d}:{:02d}".format(int(elapsed_time // 3600),
                                         int((elapsed_time % 3600) // 60),
                                         int((elapsed_time % 3600) % 60)))
        
        X_train = X[train, :]
        X_test = X[test, :]
        y_train = y[train]
        y_test = y[test]
    
        # Selección de características 
        
        if feature_selection == "mrmr":
            selection_ids = func.selection_mrmr(X_train, 
                                                y_train, 
                                                n_fts)
        if feature_selection == "wilcoxon":
            selection_ids = func.selection_wilcoxon(X_train, 
                                                    y_train, 
                                                    n_fts)
        if feature_selection == "anova":
            selection_ids = func.selection_anova(X_train, 
                                                 y_train, 
                                                 n_fts)
        if feature_selection == "rf":
            selection_ids = func.selection_rf(X_train, 
                                              y_train, 
                                              n_fts)
            
            
            
        X_trainFS = X_train[:, selection_ids]
        X_testFS = X_test[:, selection_ids]

        print(np.array(olink.columns[selection_ids]))
        Caracteristicas_cv = np.vstack((Caracteristicas_cv, olink.columns[selection_ids]))
    
    

        clf, y_pred, Scores, Params, weights = func.clf_lin(X_trainFS, 
                                                   y_train, 
                                                   X_testFS) 
        
        

        from sklearn.metrics import  classification_report
        report = classification_report(y_test, y_pred)
        print(report)
            
        # Curvas ROC
        
        # n_classes = np.unique(y).shape[0]
        # fpr = {}
        # tpr = {}
        # roc_auc = {}
        # for j in range(n_classes):
        #     fpr[j], tpr[j], _ = metrics.roc_curve(y_test, y_score[:, j], pos_label=j) #, X_testFS, y_test
        #     roc_auc[j] = metrics.auc(fpr[j], tpr[j])
            
        # interp_tpr = {}
        # for j in range(n_classes):
        #     interp_tpr[j] = np.interp(mean_fpr, fpr[j], tpr[j])
        #     interp_tpr[j][0] = 0.0
        #     tprs.append(interp_tpr[j])
        #     aucs.append(roc_auc[j])
            
            
            
        # -- AUC
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) #♣ , pos_label=2
        # AUC = metrics.auc(fpr, tpr)
        
        
        # if method == "SVM":
        #     Sets_Scores[test] = Scores
        #     Parametros.append(str([Params['C'], 
        #                            Params['gamma'], 
        #                            Params['kernel']]))
        # if method == "KNN":
        #     Parametros.append(str([Params['n_neighbors'], 
        #                            Params['p']]))
        #     # Sets_Scores[test] = np.NaN
        # if method == "LIN":
        #     Sets_Scores[test] = Scores
        #     Parametros.append(str([Params['C']]))
        
    
        tmp_confmat_aux = metrics.confusion_matrix(y_test,
                                                   y_pred,
                                                   labels=np.unique(y)).flatten()
    
    
        if 'confmat_aux' not in locals():
            confmat_aux = tmp_confmat_aux
        else:
            confmat_aux = np.vstack((confmat_aux, tmp_confmat_aux))
    
    
        print('')
        print('Iteración clasificación: ', iteracion_clf)
        acc.loc[iteracion_clf, 'Ensemble'] = round(metrics.accuracy_score(y_test, y_pred), 3)
        sens.loc[iteracion_clf, 'Ensemble'] = round(metrics.recall_score(y_test, y_pred, average='macro'), 3)
        bal.loc[iteracion_clf, 'Ensemble'] = round(metrics.balanced_accuracy_score(y_test, y_pred), 3)
        prec.loc[iteracion_clf, 'Ensemble'] = round(metrics.precision_score(y_test, y_pred, average='macro'), 3)
        f1df.loc[iteracion_clf, 'Ensemble'] = round(metrics.f1_score(y_test, y_pred, average='macro'), 3)
        print('')
        print('### --------------------------------------- ###')
        print('')
        iteracion_clf += 1
        

        
        # =============================================================================
        #         Matrices de confusión
        # =============================================================================
        
        sum_cm = np.sum(confmat_aux, axis=0).astype(np.int64).reshape(4, 4)
        plt.rcParams['font.size'] = 15
        plt.figure(figsize=(12.8, 9.6))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sum_cm,
                                              display_labels=label_encoder.inverse_transform(np.unique(y))) 
    
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        # plt.title(experimento + ' ' + str(feature_selection) + ' with ' + 
        #           classifier)
        plt.ylabel('Actual')
        plt.xlabel('Predicción')
        plt.savefig(path_resultados + 
                    experimento +
                    ' ' +
                    str(feature_selection) +
                    ' matconf ' +  
                    'n_fts=' +
                    str(n_fts) +
                    ' CV' +
                    str(splits) +
                    '.png')
        
        
        
        
        # =============================================================================
        #         Curvas ROC
        # =============================================================================       
         
        # mean_tpr = {}
        # mean_auc = {}
        # std_auc = {}
        # for j in range(n_classes):
        #     mean_tpr[j] = np.mean(tprs[j::n_classes], axis=0)
        #     mean_tpr[j][-1] = 1.0
        #     mean_auc[j] = metrics.auc(mean_fpr, mean_tpr[j])
        #     std_auc[j] = np.std(aucs[j::n_classes])
            
        # plt.figure(figsize=(8, 6))
        # for j in range(n_classes):
        #     plt.plot(mean_fpr, 
        #              mean_tpr[j], 
        #              label='{} (AUC = {:.2f})'.format(label_encoder.inverse_transform(olink['Clase'].unique())[j],  # label_encoder.inverse_transform([j])
        #              mean_auc[j]))
        #     plt.fill_between(mean_fpr, 
        #                      mean_tpr[j] - std_auc[j], 
        #                      mean_tpr[j] + std_auc[j], 
        #                      alpha=.2)
            
        # plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        # plt.xlabel('Especificidad')
        # plt.ylabel('Sensibilidad')
        # plt.ylim([0, 1.1])
        # plt.xlim([0, 1.1])
        # # plt.title('ROC Curve')
        # plt.legend(loc='lower right')
        # plt.show()
        # plt.savefig(path_resultados +
        #             experimento +
        #             ' ' +
        #             str(feature_selection) +
        #             ' ROC ' +
        #             classifier +
        #             ' n_fts=' +
        #             str(n_fts) +
        #             ' CV' +
        #             str(splits) +
        #             '.png')
        # plt.rcParams['font.size'] = 14
        
        
        
        
        
        
        
        
        
    # =============================================================================
    #    Características elegidas
    # =============================================================================
    Caracteristicas_clf = Caracteristicas_cv[Contador_Iteracion - splits*reps : Contador_Iteracion]
    caracteristicas_todo = Caracteristicas_clf.flatten()

    count = Counter(caracteristicas_todo)
    df = pd.DataFrame.from_dict(count, orient='index')
    df = df.sort_values(0, ascending=False)
    df = df.head(5)
    # plt.rcParams['font.size'] = 19
    fig_hist, ax_hist = plt.subplots(figsize=(12, 10))
    df.plot(kind='bar', legend=False, ax=ax_hist)
    # plt.title(classifier + ' with ' + feature_selection)
    plt.savefig(path_resultados + 
                experimento +
                ' ' +
                str(feature_selection) +
                ' hist' +  
                ' n_fts=' +
                str(n_fts) +
                ' CV' +
                str(splits) +
                '.png')
        
    
    caracteristicas_finales = pd.concat([caracteristicas_finales, 
                                         pd.DataFrame(df.index)],   #☺
                                         axis=1)

   
    
    # =============================================================================
    #     Resultados
    # =============================================================================
    
    
    resultados_selection = pd.DataFrame()
    resultados_selection = resultados_selection.append(acc.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(sens.mean(axis=0), 
                                                       ignore_index=True)
    # resultados_selection = resultados_selection.append(spec.mean(axis=0), ignore_index=True)
    resultados_selection = resultados_selection.append(bal.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(prec.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(f1df.mean(axis=0), 
                                                       ignore_index=True)
    # resultados_selection = resultados_selection.append(aucdf.mean(axis=0), ignore_index=True)
    # resultados_selection.loc[:, 'Metric'] = ['Accuracy', 'Sensitivity', 'Specificity', \
    #                                'Balanced Accuracy', 'Precision', 'F1', 'AUC']
    resultados_selection.loc[:, 'Metric'] = ['Accuracy', 'Sensitivity', \
                                   'Balanced Acc.', 'Precision', 'F1']        
    resultados_selection = resultados_selection.round(3) 
        
    
    
    resultados_selection.to_latex(path_resultados + 
                            experimento +
                            ' ' +
                            str(feature_selection) +
                            
                            ' ' +
                            ' resultados ' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits),
                            caption=experimento + ' ' + str(feature_selection) +
                                ' ' +
                                ' resultados ' + 
                                ' n_fts=' +
                                str(n_fts) +
                                ' CV' +
                                str(splits) +
                                '.tex',
                            index=False,
                                # header=False,
                            label=experimento + ' ' + str(feature_selection) +
                                ' resultados ' + 
                                ' n_fts=' +
                                str(n_fts) +
                                ' CV' +
                                str(splits))
  
    resultados_selection.loc[:, 'setup'] = experimento
    resultados_selection.loc[:, 'selection'] = feature_selection
    
    resultados_selection.to_csv(path_resultados + 
                            experimento +
                            ' ' +
                            str(feature_selection) +
                            
                            ' ' +
                            ' resultados ' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.csv',
                            sep = ";")
    
    resultados_finales = resultados_finales.append(resultados_selection, 
                                                   ignore_index=True)
    
    # =============================================================================
    #     Correlaciones y dendrogramas
    # =============================================================================
    
    
    # df_prov = olink.loc[:, df.index]
    # df_prov = df_prov.astype(float)
    # correlations = df_prov.corr()

    # plt.figure(figsize=(16, 12))
    # plt.rcParams['font.size'] = 19
    # dissimilarity = 1 - abs(correlations)
    # Z = linkage(squareform(dissimilarity), 'complete')
    # plt.gca().tick_params(axis='y', labelleft=False)
    # dendrogram(Z, labels=df.index, orientation='top', 
    #            leaf_rotation=90)
    
    # plt.savefig(path_resultados + 
    #             experimento +
    #             ' ' +
    #             str(feature_selection) +
    #             ' dendrogram ' +
    #             ' n_fts=' +
    #             str(n_fts) +
    #             ' CV' +
    #             str(splits) +
    #             '.png')
    

    # threshold = 0.8
    # labels = fcluster(Z, threshold, criterion='distance')

    # # labels

    # # Keep the indices to sort labels
    # labels_order = np.argsort(labels)

    # # Build a new dataframe with the sorted columns
    # for idx, i in enumerate(df_prov.columns[labels_order]):
    #     if idx == 0:
    #         clustered = pd.DataFrame(df_prov[i])
    #     else:
    #         df_to_append = pd.DataFrame(df_prov[i])
    #         clustered = pd.concat([clustered, df_to_append], axis=1)
            
    # plt.figure(figsize=(15,10))
    # correlations = clustered.corr()
    # sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
    #             annot_kws={"size": 7}, vmin=-1, vmax=1)
    # plt.savefig(path_resultados + 
    #             experimento +
    #             ' ' +
    #             str(feature_selection) +
    #             ' corr ' +
    #             ' n_fts=' +
    #             str(n_fts) +
    #             ' CV' +
    #             str(splits) +
    #             '.png')      
    

    # plt.figure(figsize=(15,10))
    # sns.clustermap(correlations, method="complete", cmap='RdBu', annot=True, 
    #                annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))
    
    # plt.savefig(path_resultados + 
    #             experimento +
    #             ' ' +
    #             str(feature_selection) +
    #             ' dendrocorr' +
    #             ' n_fts=' +
    #             str(n_fts) +
    #             ' CV' +
    #             str(splits) +
    #             '.png')
        
    

# %% Resultados finales


caracteristicas_finales.to_latex(path_resultados + 
                        experimento +
                        ' TOP5 final' + 
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits) +
                        '.tex',
                        index=False,
                        header=False,
                        caption=experimento +
                            ' TOP5 final' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits),
                        label=experimento +
                            ' TOP5 final' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits) )


resultados_finales = resultados_finales.pivot(index='Metric',
                                               columns='selection',
                                               values=['Ensemble'])


resultados_finales.to_latex(path_resultados + 
                        experimento +
                        ' resultados finales' + 
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits),
                        caption=experimento +
                            ' resultados finales' + 
                            ' n_fts=' + 
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.tex',
                        index=True,
                            # header=False,
                        label=experimento +
                            ' resultados finales ' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits))