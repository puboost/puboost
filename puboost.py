
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class PUBoostClassifier():
    def __init__(self):
        '''
        Description
        '''
        self.model_list = []
        
    def fit(self, X_seed, X_poblacion, random_state, T=50, clf='logistic', 
            l1=1, l2=1, e1=1, e2=1, **kwargs_clf):
        """
        Returns avg of oob predictions of classifier para la poblacion
        Param:
            - T number of baggint iteractions 
            - clf: base estimator (one of rg, logistic)
        """
        self.T = T
        
        # K: size of boostrap sample (= size of seed)
        K = X_seed.shape[0]
        # U: size of poblation
        U = X_poblacion.shape[0]
        # se entrena con una muestra balanceada
        # vector target: primero seed - luego poblacion
        y_poblacion = np.zeros(U)
        # y_train = np.concatenate([np.ones(K), np.zeros(K)])
        # initialize numerador de predicciones
        pred = np.zeros(U)
        # initialize denominador de predicciones
        n = np.zeros(U)
        # iniialize weight vectors
        w_poblacion = np.ones(U)
        w_seed = np.ones(K)

        # bagging
        for t in range(T):
            # get sample
            idx_train = np.random.choice(U, K, replace=True)
            X_train = np.concatenate([X_seed, X_poblacion.iloc[idx_train,:]])
            # y_train vector
            y_train = np.concatenate([np.ones(K), y_poblacion[idx_train]])
            # weights
            # print(w_poblacion[idx_train], "/n")
            weights = np.concatenate([w_seed, w_poblacion[idx_train]])      
            # train
            if clf=='rf':
                clf = RandomForestClassifier(**kwargs_clf)
            if clf=='logistic':
                clf = LogisticRegression(**kwargs_clf)
            if clf=='tree':
                clf = DecisionTreeClassifier(**kwargs_clf)
            if clf=='knn':
                clf = KNeighborsClassifier(**kwargs_clf)
            clf.fit(X_train, y_train, sample_weight = weights)
            
            self.model_list.append(clf)
            
            # predict OOB
            idx_oob = np.full(U, True)
            idx_oob[idx_train] = False
            _pred = clf.predict_proba(X_poblacion.iloc[idx_oob,:])[:,clf.classes_ == 1].ravel()
            pred[idx_oob] += _pred
            n[idx_oob] += 1
            # update weight vector
            if t > (T*l1):
                _wupdate = np.zeros(U)
                _wupdate[idx_oob] = _pred
                w_poblacion += (-_wupdate/T*l2) 
            if t > (T*e1):
                y_poblacion[(pred/n)>e2] = 1
        scores = pred / n
        return scores
        
    def predict(self, df):
        
        predic = np.zeros(df.shape[0])
        
        for t in range(self.T):
            _predic = self.model_list[t].predict_proba(df)[:,self.model_list[t].classes_ == 1].ravel()
            predic += _predic
        
        return predic / self.T