# -*- coding: utf-8 -*-
"""
Script de clasificación de las bases de datso OLINK y SOMA. Resultados baseline
(sin selección de características ni ajuste de hiperparámetros)
"""

from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
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


dataset = 'Olink'
# dataset = 'Soma'

clf = 'LIN' # LIN, KNN

# 4 clases: 'HC', 'MSA', 'PD', 'PSP'
# negative_class = "HC" 
# negative_class = "MSA"
# negative_class = "PD"
# negative_class = 'PSP'
# negative_class = "HC+PD"
negative_class = "HC+PSP"
# negative_class = "MSA+PSP"

# positive_class = "MSA" 
# positive_class = "PD"
# positive_class = "PSP"
# positive_class = "MSA+PSP"
positive_class = "MSA+PD"
# positive_class = "RESTO" 

if negative_class == 'MSA+PSP' and positive_class == 'PD':
    caso_especial = 1 # Solo para MSA+PSP vs PD
else:
    caso_especial = 0

# plt.rcParams['font.size'] = 14

if "+" not in positive_class or "+" not in negative_class: 
    list_classes = ["HC", "MSA", "PD", "PSP"]
    if caso_especial != 1:
        list_classes.remove(str(negative_class))
    
path_resultados = 'Resultados 2/' + dataset +'/'  
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

classifier = 'baseline' #+ clf
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


ids = np.arange(len(olink.columns)-1) # índices de las variables


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

caracteristicas_final = pd.DataFrame()

start_time = time.time()


# %% Validación cruzada

iteracion_clf = 0
fig_roc, ax_roc = plt.subplots(figsize=(6, 6)) # Curvas roc

confmat_aux = np.array([0, 0, 0, 0])
for train, test in split.split(X, y):
    
    
    Contador_Iteracion = Contador_Iteracion + 1
    porcentaje = Contador_Iteracion/(splits * 
                                     reps) * 100
    print('')
    print('')
    print('Completado: ' + str(Contador_Iteracion),
          '/', 
          splits * reps, 
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


        
    if clf == 'KNN':
        clf, y_pred, Scores = func.clf_knn_baseline(X_train, y_train, X_test)
    
    if clf == 'LIN':
        clf, y_pred, Scores, weights = func.clf_lin_baseline(X_train, y_train, X_test)
    
    
    coef = weights.ravel()  # Aplanar el array en caso de que sea multidimensional
    abs_coef = np.abs(coef)  # Tomar el valor absoluto de los coeficientes

    # Ordenar los índices de los coeficientes de mayor a menor
    sorted_indices = np.argsort(abs_coef)[::-1] 
    
    # Si tienes un vector de nombres de características que corresponde a los coeficientes, puedes ordenarlo también
    feature_names = np.array(olink.columns)  # Asegurarse de que es un array de numpy
    sorted_feature_names = feature_names[sorted_indices]
    
    # Inicialización curvas ROC
    viz = metrics.plot_roc_curve(clf, X_test, y_test,
                          # name='ROC fold {}'.format(Contador_Iteracion+1),
                          alpha=0.1, lw=1, ax=ax_roc)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) #♣ , pos_label=2
    AUC = metrics.auc(fpr, tpr)
    
    
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

    tn, fp, fn, tp = tmp_confmat_aux # metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y)).ravel()
    # cm = metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y)).ravel()

    print('')
    print('Iteración clasificación: ', iteracion_clf)
    acc.loc[iteracion_clf, classifier] = metrics.accuracy_score(y_test, y_pred) # acc.loc[Contador_Iteracion%(splits*reps), classifier]
    sens.loc[iteracion_clf, classifier] = metrics.recall_score(y_test, y_pred)
    spec.loc[iteracion_clf, classifier] = tn / (tn + fp)  
    bal.loc[iteracion_clf, classifier] = metrics.balanced_accuracy_score(y_test, y_pred)
    prec.loc[iteracion_clf, classifier] = metrics.precision_score(y_test, y_pred)
    f1df.loc[iteracion_clf, classifier] = metrics.f1_score(y_test, y_pred)
    aucdf.loc[iteracion_clf, classifier] = AUC            
    print('')
    print('### --------------------------------------- ###')
    print('')
    iteracion_clf += 1



# weights_ordered = np.sort(weights)
# top5_index = weights_ordered[:, 0:5]
# selected_weights = np.array(olink.columns[top5_index])

# Obtener los índices de los 5 pesos más grandes
top5_indices = np.argsort(np.abs(weights))[:, -5:]
# top5_indices = top5_indices[-5:]
# Obtener los 5 pesos más grandes y sus características correspondientes
top5_weights = weights[:, top5_indices]
top5_features = np.array(olink.columns[top5_indices])


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
plt.figure()
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sum_cm,
                                      display_labels=label_encoder.inverse_transform(np.unique(y))) 

disp.plot(cmap=plt.cm.Blues, colorbar=False)
# plt.title(experiment + ' ' + str(feature_selection) + ' with ' + 
#           classifier)
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.savefig(path_resultados + 
            experiment +
            ' ' +
            ' matconf ' +  
            str(classifier) +
            ' n_fts=' +
            str(n_fts) +
            ' CV' +
            str(splits) +
            '.png')

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

# plt.figure()
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
        label='Chance', alpha=.7)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax_roc.plot(mean_fpr, mean_tpr, color='green',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]) #, title="Receiver operating characteristic example"
ax_roc.legend(loc="lower right")
ax_roc.get_legend().remove()

# Here is the trick
# ax_roc.gcf()
handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[6:]
labels = labels[6:]
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.legend('off')
# ax_roc.set_title(experiment + ' for ' + classifier + ' with ' + feature_selection)
# ax_roc.show()
ax_roc.set_xlabel('Especificidad')
ax_roc.set_ylabel('Sensitividad')
ax_roc.figure.savefig(path_resultados + 
            experiment +
            ' ROC ' +  
            str(classifier) +
            ' n_fts=' +
            str(n_fts) +
            ' CV' +
            str(splits) +
            '.png')


Caracteristicas_clf = Caracteristicas_cv[Contador_Iteracion - splits*reps : Contador_Iteracion]


        
        

    
# =============================================================================
#     Resultados
# =============================================================================


resultados_selection = pd.DataFrame()
resultados_selection = resultados_selection.append(acc.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(sens.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(spec.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(bal.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(prec.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(f1df.mean(axis=0), ignore_index=True)
resultados_selection = resultados_selection.append(aucdf.mean(axis=0), ignore_index=True)
resultados_selection.loc[:, 'Metric'] = ['Accuracy', 'Sensitivity', 'Specificity', \
                               'Balanced Acc.', 'Precision', 'F1', 'AUC']
        
resultados_selection.round(2) 
    


resultados_selection.to_latex(path_resultados + 
                        experiment +
                        ' resultados ' + 
                        classifier +
                        ' n_fts=' +
                        str(n_fts) +
                        ' CV' +
                        str(splits),
                        caption=experiment +
                            ' resultados ' + 
                            classifier + 
                            ' n_fts=' + 
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.tex',
                        index=False,
                            # header=False,
                        label=experiment +
                            ' resultados ' + 
                            classifier + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits))

  
resultados_selection.loc[:, 'setup'] = experiment
resultados_selection.loc[:, 'selection'] = 'baseline'

resultados_selection.to_csv(path_resultados + 
                            experiment +
                            ' resultados ' + 
                            classifier + 
                            ' n_fts=' +
                            str(n_fts) +
                            ' CV' +
                            str(splits) +
                            '.csv',
                            sep = ";")

 

    
 
    




