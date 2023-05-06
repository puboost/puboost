import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def bagged_clf(X_seed, X_poblacion, random_state, T=50, clf='rf', feature_importance=True, **kwargs_clf):
    """
    Returns avg of oob predictions of classifier para la poblacion
    Param:
        - T number of baggint iteractions 
        - clf: base estimator (one of rg, logistic)
    """
    # K: size of boostrap sample (= size of seed)
    K = X_seed.shape[0]
    # U: size of poblation
    U = X_poblacion.shape[0]
    # se entrena con una muestra balanceada
    # vector target: primero seed - luego poblacion
    y_train = np.concatenate([np.ones(K), np.zeros(K)])
    # initialize numerador de predicciones
    pred = np.zeros(U)
    # initialize denominador de predicciones
    n = np.zeros(U)
    # initialize feature importance
    importance = np.zeros(X_seed.shape[1])

    # bagging
    for t in range(T):
        # get sample
        idx_train = np.random.choice(U, K, replace=True)
        X_train = np.concatenate([X_seed, X_poblacion.iloc[idx_train,:]])
        # train
        if clf=='rf':
            clf = RandomForestClassifier(**kwargs_clf)
        if clf=='logistic':
            clf = LogisticRegression(**kwargs_clf)
        if clf=='tree':
            clf = DecisionTreeClassifier(**kwargs_clf)
        if clf=='knn':
            clf = KNeighborsClassifier(**kwargs_clf)
        clf.fit(X_train, y_train)
        # predict OOB
        idx_oob = np.full(U, True)
        idx_oob[idx_train] = False
        _pred = clf.predict_proba(X_poblacion.iloc[idx_oob,:])[:,clf.classes_ == 1].ravel()
        pred[idx_oob] += _pred
        n[idx_oob] += 1
        importance += clf.feature_importances_
    scores = pred / n
    if feature_importance:
        feat_importance = importance / T
        return scores, feat_importance
    else:
       return scores