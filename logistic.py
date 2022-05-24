import math
from math import e
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class CostFunction:
    
    def __init__(self):
        pass
    
    def mse(self, Y, pred_y, returnList=False):
        if returnList:
            return np.array([(pred_y[i]-y)**2 for i, y in enumerate(Y)])
        else:
            return sum([(pred_y[i]-y)**2 for i, y in enumerate(Y)])/len(Y)
    
    def logloss(self, Y, pred_y, returnList=False):
        Y, pred_y = Y.tolist(), pred_y.tolist()
        if returnList:
            return np.array([y * math.log(pred_y[i]) + (1-y) * math.log(1-pred_y[i]) for i, y in enumerate(Y)])
        else:
            return sum([y * math.log(pred_y[i]) + (1-y) * math.log(1-pred_y[i]) for i, y in enumerate(Y)])/(-len(Y))

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=1000):

        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.scaler = StandardScaler()
        self.accuracy = 0
        self.firstErrors = []
        self.lastErrors = []

    def set_features(self, features):
        self.features = features


    def standardize(self):
        self.features = self.scaler.fit_transform(self.features)
        self.features = np.insert(self.features, 0, 1, axis=1)

    def traintest_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, train_size=0.8, random_state=43)

    def sigmoid(self, t):
        return 1/(1+e**(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_logloss(self):
        coef_ = self.coef_
        N = len(self.y_train.tolist())
        for _ in range(self.n_epoch):
            for i, row in enumerate(self.X_train):

                y_hat = self.predict_proba(row, self.coef_)
                realY = self.y_train[i]

                for j, X in enumerate(row):
                    self.coef_[j] -= (self.l_rate * (y_hat - realY) * X) / N

                if _ == 0:
                    self.firstErrors.append( ((realY*math.log(y_hat)) + ((1-realY) * math.log(1-y_hat))) / (-N))
                elif _ == 999:
                    self.lastErrors.append( ((realY*math.log(y_hat)) + ((1-realY) * math.log(1-y_hat))) / (-N))



    def fit_mse(self):
        coef_ = self.coef_
        N = len(self.y_train.tolist())
        for _ in range(self.n_epoch):
            for i, row in enumerate(self.X_train):

                y_hat = self.predict_proba(row, self.coef_)
                realY = self.y_train[i]

                for j, X in enumerate(row):
                    deriv = self.l_rate * ((y_hat - realY) * y_hat * (1-y_hat) * X)
                    self.coef_[j] = self.coef_[j] - (deriv)

                if _ == 0:
                    self.firstErrors.append(((realY - y_hat)**2)/N)
                elif _ == 999:
                    self.lastErrors.append(((realY - y_hat)**2)/N)


    def fit_skl(self):
        model = LogisticRegression(fit_intercept = True)
        model.fit(self.X_train, self.y_train)
        self.coef_ = np.append(model.intercept_, model.coef_)
        self.coef_ = model.coef_.flatten()

    def predict(self, coef_, X_test, y_test, cut_off=0.5):
        if cut_off:
            predictions = [int(self.predict_proba(row, coef_)>=cut_off) for row in X_test]
            y_test = y_test.tolist()
            self.y_test = y_test
            if y_test:
                num_right = 0
                for i, pred in enumerate(predictions):
                    if pred == y_test[i]:
                        num_right += 1

                self.accuracy = num_right/len(y_test)
            self.y_test = np.array(self.y_test)
        else:
             predictions = [self.predict_proba(row, coef_) for row in X_test]

        return np.array(predictions)

    def reset(self):
        self.coef_ = np.array([0.0 for i in range(self.num_of_features+1)])
        self.accuracy = 0
        self.y_test = np.array(self.y_test)


def new(n_epoch):
    regr = CustomLogisticRegression(n_epoch)
    
    # grab data
    regr.data = sklearn.datasets.load_breast_cancer()
    regr.target = regr.data['target']
    regr.data = pd.DataFrame(regr.data.data, columns=[regr.data.feature_names])
    regr.set_features(np.array(regr.data[["worst concave points", "worst perimeter", "worst radius"]]))
    
    # set initial coefficients
    regr.num_of_features = 3
    regr.coef_ = np.array([0.0 for i in range(regr.num_of_features+1)])
    
    # data preparation
    regr.standardize()
    regr.traintest_split()
    
    return regr

# train
cost = CostFunction()

#logloss
logloss = new(n_epoch=1000)
logloss.fit_logloss()
loglossCoef = logloss.coef_
loglossFirstEs = logloss.firstErrors
loglossLastEs = logloss.lastErrors

loglossPreds = logloss.predict(loglossCoef, logloss.X_test, logloss.y_test)
loglossAcc = logloss.accuracy




# mse
mse = new(n_epoch=1000)
mse.fit_mse()
mseCoef = mse.coef_

mseFirstEs = mse.firstErrors
mseLastEs = mse.lastErrors

msePreds = mse.predict(mseCoef, mse.X_test, mse.y_test)
mseAcc = mse.accuracy



#sklearn
skl = new(n_epoch=1000)
skl.fit_skl()
sklCoef = skl.coef_
sklPreds = skl.predict(sklCoef, skl.X_test, skl.y_test)


final_dict = {'mse_accuracy': mse.accuracy, 'logloss_accuracy': logloss.accuracy, 'sklearn_accuracy': skl.accuracy, 'mse_error_first': mseFirstEs, 'mse_error_last': mseLastEs, 'logloss_error_first': loglossFirstEs, 'logloss_error_last': loglossLastEs}
print(final_dict)


print(f"""Answers to the questions:
1) 0.00003
2) 0.00000
3) {str(round(max(loglossFirstEs), 5))}
4) {str(round(max(loglossLastEs), 5))}
5) expanded
6) expanded""")

