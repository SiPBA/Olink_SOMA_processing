# -*- coding: utf-8 -*-
"""
Script de clasificación de las bases de datos OLINK y SOMA.
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
from imblearn.under_sampling import CondensedNearestNeighbour
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
under = True # Undersampling para cuando Resto
fts_selection = ['anova', 'mrmr', 'rf', 'wilcoxon']
classifiers = ['SVM', 'LIN', 'KNN', 'DT']


dataset = 'Olink'
# dataset = 'Soma'

# 4 clases: 'HC', 'MSA', 'PD', 'PSP'
negative_class = "HC" 
# negative_class = "MSA"
# negative_class = "PD"
# negative_class = 'PSP'
# negative_class = "HC+PD"
# negative_class = "MSA+PD"
# negative_class = "MSA+PSP"

# positive_class = "MSA" 
# positive_class = "PD"
# positive_class = "PSP"
# positive_class = "MSA+PSP"
# positive_class = "HC+PSP"
positive_class = "RESTO" 


if negative_class == 'MSA+PSP' and positive_class == 'PD':
    caso_especial = 1 # Solo para MSA+PSP vs PD
else:
    caso_especial = 0

plt.rcParams['font.size'] = 14

if "+" not in positive_class or "+" not in negative_class: 
    list_classes = ["HC", "MSA", "PD", "PSP"]
    if caso_especial != 1:
        list_classes.remove(str(negative_class))
    
    
path_resultados = 'Resultados ' + dataset +'/'  

# =============================================================================
# # Lectura de datos OLINK
# =============================================================================

if dataset == 'Olink':
    olink = pd.read_csv('olink.csv')
    olink = olink[olink["SampleID"].str.contains("PD - 5") == False]
    olink = olink.drop(columns=['SampleID', 'Unnamed: 0'])
    
if dataset == 'Soma':
    olink = pd.read_csv('soma.csv')
    olink = olink[olink["SampleID"].str.contains("PD - 5") == False]
    olink = olink.drop(columns=['Unnamed: 0', 'SampleID'])
    olink = olink.replace('. ','.', regex=True)


if positive_class == "RESTO":
    olink = olink.replace(list_classes, 'RESTO')
elif "+" in positive_class or "+" in negative_class:
    neg = negative_class.split("+", 2)
    first = neg[0]
    second = neg[1]
    olink = olink.replace([first, second], negative_class)
    
    if caso_especial !=1:
        pos = positive_class.split("+", 2)
        third = pos[0]
        fourth = pos[1]
        olink = olink.replace([third, fourth], positive_class) 
    if caso_especial == 1:
        olink.drop(olink[olink['Class'] == 'HC'].index, inplace=True)
else:
    olink = olink.loc[(olink['Class'] == negative_class) | (olink['Class'] == positive_class)]



olink['Clase'] = olink['Class']
olink = olink.drop(columns=['Class'])

experiment = negative_class + ' vs ' + positive_class
path_resultados += experiment + '/'


# Discretización de la variable Class

data = olink.loc[:, 'Clase']
values = np.array(data)

label_encoder = LabelEncoder()
encoded = label_encoder.fit_transform(values)
le_name_mapping = dict(zip(label_encoder.classes_, 
                           label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

olink['Clase'] = encoded
nombres = olink.columns



# Generación de los conjuntos y etiquetas 


X = olink.drop(columns=['Clase'])
y = olink.loc[:, 'Clase']

X = np.asarray(X, dtype=float) 
y = np.array(y)


# Undersampling
if under and positive_class == "RESTO":
    
    
    undersample = CondensedNearestNeighbour(sampling_strategy='majority', n_neighbors=8)
    X, y = undersample.fit_resample(X, y)
    Counter(y)



# Validación cruzada 

split = RepeatedStratifiedKFold(n_splits = splits, 
                                   n_repeats = reps, 
                                   random_state = None) 

# Variables de salida 

Sets_Scores = np.zeros_like(y, dtype=float) # Puntuaciones
Parametros = [] # Parámetros de clasificación
Caracteristicas_cv = np.nan * np.ones([0, n_fts]) # Matriz 
Contador_Iteracion = 0

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []


acc = pd.DataFrame()
sens = pd.DataFrame()
spec = pd.DataFrame()
bal = pd.DataFrame()
prec = pd.DataFrame()
f1df = pd.DataFrame()
aucdf = pd.DataFrame()
resultados = pd.DataFrame()
resultados_finales = pd.DataFrame()

caracteristicas_finales = pd.DataFrame()

start_time = time.time()


# %% Validación cruzada

for i, feature_selection in enumerate(fts_selection):
    
    for j, classifier in enumerate(classifiers):
      
        iteracion_clf = 0
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6)) # Curvas ROC
        
        confmat_aux = np.array([0, 0, 0, 0])
        for train, test in split.split(X, y):
            
            
            Contador_Iteracion = Contador_Iteracion + 1
            porcentaje = Contador_Iteracion/(splits * 
                                             reps * 
                                             len(classifiers) * 
                                             len(fts_selection)) * 100
            print('Selección: ' + feature_selection)
            print('')
            print('Clasificador: ' + classifier)
            print('')
            print('Completado: ' + str(Contador_Iteracion),
                  '/', 
                  splits * reps * len(classifiers) * len(fts_selection), 
                  '(',
                  round(porcentaje, 2),
                  '% )') # Lleva la cuenta de la simulación
            
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
        
        
            # Clasificación 
            if classifier == "SVM": # SVM
                clf, y_pred, Scores, Params = func.clf_svm(X_trainFS, 
                                                           y_train, 
                                                           X_testFS) 
            if classifier == "KNN": # KNN
                clf, y_pred, Scores, Params = func.clf_knn(X_trainFS, 
                                                           y_train, 
                                                           X_testFS) 
            if classifier == "DT": # DT
                clf, y_pred = func.clf_dt(X_trainFS, 
                                          y_train, 
                                          X_testFS) 
            if classifier == "LIN": # LIN
                clf, y_pred, Scores, Params = func.clf_lin(X_trainFS, 
                                                           y_train, 
                                                           X_testFS) 
                
            
            
            # Inicialización curvas ROC
            viz = metrics.plot_roc_curve(clf, X_testFS, y_test,
                                  # name='ROC fold {}'.format(Contador_Iteracion+1),
                                  alpha=0.1, lw=1, ax=ax_roc)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) #♣ 
            AUC = metrics.auc(fpr, tpr)
            
    
            tmp_confmat_aux = metrics.confusion_matrix(y_test,
                                                       y_pred,
                                                       labels=np.unique(y)).flatten()
        
        
            if 'confmat_aux' not in locals():
                confmat_aux = tmp_confmat_aux
            else:
                confmat_aux = np.vstack((confmat_aux, tmp_confmat_aux))
        
            tn, fp, fn, tp = tmp_confmat_aux 
        
            print('')
            print('Iteración clasificación: ', iteracion_clf)
            acc.loc[iteracion_clf, classifier] = round(metrics.accuracy_score(y_test, y_pred), 3)
            sens.loc[iteracion_clf, classifier] = round(metrics.recall_score(y_test, y_pred), 3)
            spec.loc[iteracion_clf, classifier] = round(tn / (tn + fp), 3)
            bal.loc[iteracion_clf, classifier] = round(metrics.balanced_accuracy_score(y_test, y_pred), 3)
            prec.loc[iteracion_clf, classifier] = round(metrics.precision_score(y_test, y_pred), 3)
            f1df.loc[iteracion_clf, classifier] = round(metrics.f1_score(y_test, y_pred), 3)
            aucdf.loc[iteracion_clf, classifier] = round(AUC)        
            print('')
            print('### --------------------------------------- ###')
            print('')
            iteracion_clf += 1
        
        
        
        # =============================================================================
        #         Matrices de confusión
        # =============================================================================
        sum_cm = np.sum(confmat_aux, axis=0).astype(np.int64).reshape(2, 2)
        # sum_cm = np.asmatrix(sum_cm)
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=meanCM,
        #                        display_labels=np.unique(y))
        # disp.plot() 
        # disp = metrics.plot_confusion_matrix(clf, X_testFS, y_test,
        #                          display_labels=label_encoder.inverse_transform(np.unique(y)),
        #                          cmap=plt.cm.Blues,
        #                          normalize=None,
        #                          colorbar=False)
        # ax.show() 
        # plt.rcParams['font.size'] = 15
        plt.figure(figsize=(12.8, 9.6))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sum_cm,
                                              display_labels=label_encoder.inverse_transform(np.unique(y))) 
    
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        # plt.title(experiment + ' ' + str(feature_selection) + ' with ' + 
        #           classifier)
        plt.ylabel('Actual') # , fontsize=18
        plt.xlabel('Predicción')
        plt.savefig(path_resultados + 
                    experiment +
                    ' ' +
                    str(feature_selection) +
                    ' matconf ' +  
                    str(classifier) +
                    ' n_fts=' +
                    str(n_fts) +
                    ' CV' +
                    str(splits) +
                    '.png')
        # plt.rcParams['font.size'] = 14
        
        # MatConf normalizada
        # min_value = np.min(sum_cm)
        # max_value = np.max(sum_cm)
        # sum_cm_norm = sum_cm / (splits*reps*3) # splits*reps*X.shape[0]   3=6/2 siendo 6 el num de sujetos evaluados y 2 el num de clases
        # plt.figure()
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sum_cm_norm,
        #                                       display_labels=label_encoder.inverse_transform(np.unique(y))) 
    
        # disp.plot(cmap=plt.cm.Blues, colorbar=False)
        # # plt.title(experiment + ' ' + str(feature_selection) + ' with ' + 
        # #           classifier)
        # plt.ylabel('Actual')
        # plt.xlabel('Predicción')
        # plt.savefig(path_resultados + 
        #             experiment +
        #             ' ' +
        #             str(feature_selection) +
        #             ' ' +
        #             ' matconf ' +  
        #             str(classifier) +
        #             ' norm n_fts=' +
        #             str(n_fts) +
        #             ' CV' +
        #             str(splits) +
        #             '.png')
        
        
        
        # =============================================================================
        #         Curvas ROC
        # =============================================================================       
        
        # plt.figure(figsize=(6, 12))
        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
                # label='{} (AUC = {:.2f})'.format()[j] AUC
                # label='{} (AUC = {:.2f})'.format(),
                # label='Chance', 
                alpha=.7)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax_roc.plot(mean_fpr, mean_tpr, color='green',
                label=r'ROC promedio (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, 
                alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ Desv. est.')

        ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]) #, title="..."
        handles, labels = ax_roc.get_legend_handles_labels()

        ax_roc.legend(handles[-2:], labels[-2:], loc="lower right")
        by_label = dict(zip(labels, handles))

        
        ax_roc.set_xlabel('Especificidad', fontsize=19)
        ax_roc.set_ylabel('Sensibilidad', fontsize=19)
        ax_roc.figure.savefig(path_resultados + 
                    experiment +
                    ' ' +
                    str(feature_selection) +
                    ' ROC ' +  
                    str(classifier) +
                    ' n_fts=' +
                    str(n_fts) +
                    ' CV' +
                    str(splits) +
                    '.png')
        
        
        
    
        
        
    # =============================================================================
    #    Características elegidas
    # =============================================================================
    Caracteristicas_clf = Caracteristicas_cv[Contador_Iteracion - splits*reps : Contador_Iteracion]
    caracteristicas_todo = Caracteristicas_clf.flatten()

    count = Counter(caracteristicas_todo)
    df = pd.DataFrame.from_dict(count, orient='index')
    df = df.sort_values(0, ascending=False)
    df = df.head(5)
    plt.figure(figsize=(16, 12))
    # plt.rcParams['font.size'] = 19
    fig_hist, ax_hist = plt.subplots(figsize=(12, 10))
    df.plot(kind='bar', legend=False, ax=ax_hist)
    # plt.title(negative_class + ' vs ' + positive_class + ' with ' + feature_selection)
    plt.savefig(path_resultados + 
                experiment +
                ' ' +
                str(feature_selection) +
                ' hist' +  
                ' n_fts=' +
                str(n_fts) +
                ' CV' +
                str(splits) +
                '.png')
        
    # pd.DataFrame(np.array(df.index)).to_csv(path_resultados + 
    #                         experiment +
    #                         ' ' +
    #                         str(feature_selection) +
    #                         ' TOP5 ' + 
    #                         ' n_fts=' +
    #                         str(n_fts) +
    #                         ' CV' +
    #                         str(splits) +
    #                         '.csv',
    #                         sep = ";",
    #                         index=False,
    #                         header=False)
    
    # pd.DataFrame(np.array(df.index)).to_latex(path_resultados + 
    #                         experiment +
    #                         ' ' +
    #                         str(feature_selection) +
    #                         ' TOP5 ' + 
    #                         ' n_fts=' +
    #                         str(n_fts) +
    #                         ' CV' +
    #                         str(splits) +
    #                         '.tex',
    #                         index=False,
    #                         header=False,
    #                         caption=experiment +
    #                             ' ' +
    #                             str(feature_selection) +
    #                             ' TOP5 ' + 
    #                             ' n_fts=' +
    #                             str(n_fts) +
    #                             ' CV' +
    #                             str(splits),
    #                         label=experiment +
    #                             ' ' +
    #                             str(feature_selection) +
    #                             ' TOP5 ' + 
    #                             ' n_fts=' +
    #                             str(n_fts) +
    #                             ' CV' +
    #                             str(splits) )
    

    
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
    resultados_selection = resultados_selection.append(spec.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(bal.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(prec.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(f1df.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection = resultados_selection.append(aucdf.mean(axis=0), 
                                                       ignore_index=True)
    resultados_selection.loc[:, 'Metric'] = ['Accuracy', 'Sensibilidad', 'Especificidad', \
                                   'Balanced Acc.', 'Precisión', 'F1', 'AUC']
            
    resultados_selection = resultados_selection.round(3) 
        
    
    

    
  
    resultados_selection.loc[:, 'setup'] = experiment
    resultados_selection.loc[:, 'selection'] = feature_selection
    
    
    resultados_finales = resultados_finales.append(resultados_selection, 
                                                   ignore_index=True)
    
    
    # =============================================================================
    #     Correlaciones y dendrogramas
    # =============================================================================
    

    df_prov = olink.loc[:, df.index]
    df_prov = df_prov.astype(float)
    
    # Correlaciones
    correlations = df_prov.corr()
    plt.figure(figsize=(16, 12))
    plt.rcParams['font.size'] = 19
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')
    plt.gca().tick_params(axis='y', labelleft=False)
    dendrogram(Z, labels=df.index, orientation='top', 
                leaf_rotation=90)
    
    plt.savefig(path_resultados + 
                experiment +
                ' ' +
                str(feature_selection) +
                ' dendrogram' +
                ' n_fts=' +
                str(n_fts) +
                ' CV' +
                str(splits) +
                '.png')
    
    # Clusterización
    threshold = 0.8
    labels = fcluster(Z, threshold, criterion='distance')

    # labels

    # Ordenar etiquetas
    labels_order = np.argsort(labels)

    # Nuevo df con etiquetas ordenadas
    for idx, i in enumerate(df_prov.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(df_prov[i])
        else:
            df_to_append = pd.DataFrame(df_prov[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)
            
    # Cluster de correlaciones
    plt.figure(figsize=(15, 10))
    correlations = clustered.corr()
    sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    plt.savefig(path_resultados + 
                experiment +
                ' ' +
                str(feature_selection) +
                ' corr' +
                ' n_fts=' +
                str(n_fts) +
                ' CV' +
                str(splits) +
                '.png')      
    

    plt.figure(figsize=(15, 10))
    sns.clustermap(correlations, method="complete", cmap='RdBu', annot=True, 
                    annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12))
    
    plt.savefig(path_resultados + 
                experiment +
                ' ' +
                str(feature_selection) +
                ' dendrocorr' +
                ' n_fts=' +
                str(n_fts) +
                ' CV' +
                str(splits) +
                '.png')
        
    
    
    
    
    
# %% Resultados finales


nuevo_orden = ['selection', 'setup', 'Metric', 'SVM', 'LIN', 'KNN', 'DT']
resultados_finales = resultados_finales[nuevo_orden]

resultados_finales.to_latex(path_resultados + 
                        experiment +
                        ' resultados finales' + 
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits),
                        caption=experiment +
                            ' resultados finales' + 
                            ' n_fts=' + 
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.tex',
                        index=False,
                            # header=False,
                        label=experiment +
                            ' resultados finales ' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits))

caracteristicas_finales.to_latex(path_resultados + 
                        experiment +
                        ' TOP5 final' + 
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits) +
                        '.tex',
                        index=False,
                        header=False,
                        caption=experiment +
                            ' TOP5 final' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits),
                        label=experiment +
                            ' TOP5 final' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits) )


resultados_finales2 = resultados_finales.drop(columns=['setup', 'selection'])


resultados_finales2.to_latex(path_resultados + 
                        experiment +
                        ' resultados finales 2' + 
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits),
                        caption=experiment +
                            ' resultados finales' + 
                            ' n_fts=' + 
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.tex',
                        index=False,
                            # header=False,
                        label=experiment +
                            ' resultados finales ' + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits))




