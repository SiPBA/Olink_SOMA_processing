# -*- coding: utf-8 -*-
"""
Librería que contiene funciones para clasificación y análisis.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, svm, tree
from sklearn.model_selection import GridSearchCV         
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from mrmr import mrmr_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

#%% FUNCIONES %%#

def selection_wilcoxon(X_train, y_train, n_fts):
    """
    Función para seleccionar las caracteristicas más relevantes con el test de 
    Mann-Whitney-Wilcoxon.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    y_train : Array of int64
        etiquetas de entrenamiento.
    n_fts : int32
        número de características a seleccionar.
    IDs : Array of int32
        índices de las variables.

    Returns
    -------
    X_train : float64
        datos de entrenamiento tras selección.
    X_test : float64
        datos de test tras selección.
    selected_features : Array of int32
        índices de las variables.
    PVal_NonDiscarded : Array of float64
        pvalores de las variables.

    """

    # L_NumFeatures = np.size(X_train, axis=0) # Tantas vars. como instancias
    # L_NumFeatures = n_fts + 1 
    # IDs = np.arange(X_train.shape[1]-1) 
    
    # # Mann-Whitney-Wilcoxon U-Test:
    # pvalues = []
    # statistics = []
    # for col in np.arange(np.size(X_train, axis=1)):
    #     dataX = X_train[:, col]
    #     dataY = y_train
    #     res_statistic, res_pvalue = stats.mannwhitneyu(dataX, dataY)
    #     pvalues.append(res_pvalue)
    #     statistics.append(res_statistic)
    # pvalues = np.array(pvalues) # Array de los pvalores
    # orden = np.argsort(pvalues) # Indices de los pvalores ordenados de + a -

    # orden = orden[0:L_NumFeatures-1] # Selección
    # selected_features = IDs[orden]
    
    
    
    
    # PVal_NonDiscarded = pvalues[orden]
    # selectedpvalues = np.concatenate(np.where(pvalues < 0.0000001))



    # feature_scores = []
    # pvalues = []
    # statistics = []
    # i = 0
    # for col in np.arange(np.size(X_train, axis=1)):
    #     dataX = X_train[:, col]
    #     dataY = y_train
    #     statistic, p_value = stats.mannwhitneyu(dataX, dataY)
    #     feature_scores.append((i, p_value))
    #     i += 1
    #     pvalues.append(pvalues)
    #     statistics.append(statistic)

    # # Sort feature scores based on test statistic (ascending order)
    # feature_scores.sort(key=lambda x: x[1])
    
    # # Select top two features
    # selected_features = [idx for idx, _ in feature_scores[:n_fts]]
    # selected_features = np.array(selected_features)
    
    # 1. Compute Mann-Whitney-Wilcoxon U-Test for each feature
    p_values = []
    for i in range(X_train.shape[1]):
        # Separate the feature by class
        feature_class_0 = X_train[y_train == 0, i]
        feature_class_1 = X_train[y_train == 1, i]
        # Compute test statistic
        _, p_value = stats.mannwhitneyu(feature_class_0, feature_class_1, alternative='two-sided')
        p_values.append(p_value)

    # 2. Sort the features by the p-values
    sorted_indices = np.argsort(p_values)

    # 3. Select the top features
    selected_features = sorted_indices[:n_fts]

    return selected_features



def selection_mrmr(X_train, y_train, n_fts):
    """
    Selección de features con mRMR (Maximum Relevance — Minimum Redundancy).

    Parameters
    ----------
    X_train : Array float64
        Conjunto de entrenamiento.
    y_train : Array float64
        Etiquetas de entrenamiento.
    n_fts : int32
        número de características a seleccionar.

    Returns
    -------
    selected_features : ndarray
        array que contiene los índices de características seleccionadas.


    """
    
    selected_features = np.array(mrmr_classif(X=pd.DataFrame(X_train), 
                                      y=pd.DataFrame(y_train),
                                      K=n_fts,
                                      relevance='f',
                                      show_progress=False))

    # df = pd.concat([pd.DataFrame(y_train) , pd.DataFrame(X_train)], axis=1)
    # df.columns = nombres
     
    
    return selected_features




def selection_anova(X_train, y_train, n_fts):
    """
    Selección de características con ANOVA (ANalysis Of VAriance).

    Parameters
    ----------
    X_train : Array float64
        Conjunto de entrenamiento.
    y_train : Array float64
        Etiquetas de entrenamiento.
    n_fts : int32
        número de características a seleccionar.

    Returns
    -------
    selected_features : ndarray
        array que contiene los índices de características seleccionadas.


    """
    fvalue_Best = SelectKBest(f_classif, k=n_fts)
    
    selected_features = fvalue_Best.fit_transform(X_train, y_train)
    selected_features = fvalue_Best.get_support(indices=True) 


    return selected_features


def selection_rf(X_train, y_train, n_fts):
    """
    Selección de características con RF (Random Forest).

    Parameters
    ----------
    X_train : Array float64
        Conjunto de entrenamiento.
    y_train : Array float64
        Etiquetas de entrenamiento.
    n_fts : int32
        número de características a seleccionar.

    Returns
    -------
    selected_features : ndarray
        array que contiene los índices de características seleccionadas.


    """
    sel = RandomForestClassifier(n_estimators = 100)
    sel.fit(X_train, y_train)
    feature_scores = pd.Series(sel.feature_importances_).sort_values(ascending=False)
    selected_features = feature_scores.index.values.astype(int)[0:n_fts]
    
    return selected_features




def clf_svm(X_train, Y_train, X_test):
    """
    Bloque de clasificación Support Vector Machine con kernel RBF.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.
    params : dict
        mejores parámetros que se han obtenido.


    """
    
    param_grid = {'C': [1, 10, 100],
                'gamma': [1, 0.01, 'scale'],
                'kernel': ['rbf']}
    
    classifier = svm.SVC(class_weight='balanced', probability=True)
    
    gs = GridSearchCV(classifier, param_grid, cv=10)
    
    clf = make_pipeline(preprocessing.StandardScaler(), 
                        gs)
    
    
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    Scores = clf.decision_function(X_test)
    params = gs.best_params_
    
 
    return clf, Y_predict, Scores, params


def clf_knn(X_train, Y_train, X_test):
    """
    Clasificación mediante K Nearest Neighbors.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.
    params : dict
        mejores parámetros que se han obtenido.


    """
    
    k_range = list(range(1,10))
    
    param_grid = dict(n_neighbors=k_range, p=[1,2])
    classifier = KNeighborsClassifier()
    gs=GridSearchCV(classifier, param_grid,cv=10)
    
    clf = make_pipeline(preprocessing.StandardScaler(), 
                        gs)

    
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    params = gs.best_params_
    
    Scores = clf.predict_proba(X_test)


    return clf, Y_predict, Scores, params



def clf_dt(X_train,Y_train,X_test):
    """
    Clasificación mediante Decision Tree.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.
    params : dict
        mejores parámetros que se han obtenido.


    """
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)

    return clf, Y_predict


def clf_lin(X_train,Y_train,X_test):
    """
    Bloque de clasificación Support Vector Machine con kernel lineal.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.
    params : dict
        mejores parámetros que se han obtenido.


    """
    
    L_iter = 1000 

    param_grid={'C': [1, 10, 100]}
    
    classifier=svm.LinearSVC(class_weight='balanced', 
                        max_iter=L_iter,
                        dual=False)
    
    gs = GridSearchCV(classifier, param_grid, cv=10)
    
    clf = make_pipeline(preprocessing.StandardScaler(), 
                        gs)
    
    
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    Scores = clf.decision_function(X_test)
    params = gs.best_params_
    
    # Access the best LinearSVC model from the pipeline
    best_linearsvc_model = clf.named_steps['gridsearchcv'].best_estimator_
    
    # Extract the coefficient weights
    weights = best_linearsvc_model.coef_
    
    # best_svm_model = gs.best_estimator_['svm']
    
    # weights = clf.named_steps['gridsearch'].coef_
    # print(clf.named_steps)
    
    return clf, Y_predict, Scores, params, weights




def clf_lin_baseline(X_train, Y_train, X_test):
    """
    Bloque de clasificación Support Vector Machine con 
    comportamiento baseline y con kernel lineal. Para utilizar 
    sin selección de características a modo de comparación.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.

    """
    clf = svm.LinearSVC(class_weight='balanced')
    
    
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    Scores = clf.decision_function(X_test)
    weights = clf.coef_


    return clf, Y_predict, Scores, weights



def clf_knn_baseline(X_train, Y_train, X_test):
    """
    Bloque de clasificación KNN con 
    comportamiento baseline. Para utilizar 
    sin selección de características a modo de comparación.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuaciones del clasificador.

    """
    
    clf = KNeighborsClassifier()
 
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    
    Scores = clf.predict_proba(X_test)


    return clf, Y_predict, Scores





