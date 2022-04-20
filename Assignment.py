# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show, boxplot, xticks, ylabel
from matplotlib.pyplot import (imshow, ylabel, title, colorbar, cm)
from scipy.linalg import svd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, KFold
from toolbox_02450 import train_neural_net, feature_selector_lr, bmplot, mcnemar
import torch
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import itertools

# import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

file_path = 'Data'

# Gets attribute names
attribute_names = []
with open(file_path+'/glass.names', "r") as filestream:
    lineNumber = 0
    for line in filestream:
        lineNumber+=1
        if(lineNumber < 45):
            continue
        if(lineNumber > 56):
            break
        currentline = line[6:]
        attr = currentline.split(':')
        if(len(attr[0]) > 15):
            continue
        attribute_names.append(attr[0])


# Gets class names
classDict = {}
with open(file_path+'/glass.names', "r") as filestream:
    lineNumber = 0
    for line in filestream:
        lineNumber+=1
        if(lineNumber < 57):
            continue
        if(lineNumber > 63):
            break
        currentline = line[9:]
        classInfo = currentline.split(' ')
        classDict[classInfo[0]] = classInfo[1]

# Reads data
X = pd.read_csv(file_path+'/glass.data', sep=',', names=attribute_names)

# Clean up data
y = X['Type of glass']
X.drop('Type of glass', inplace=True, axis=1)
X.drop('Id number', inplace=True, axis=1)
del attribute_names[0]
del attribute_names[-1]
N = len(y)
M = len(attribute_names)
C = len(classDict)

# Data transformation
Xt = (X-X.mean())/X.std()
# np.mean(Xt).sum()/9
# Xt.std()

"""
# Compute SVD and variance
U,S,Vh = svd(Xt,full_matrices=False)`
V = Vh.transpose()
rho = (S*S) / (S*S).sum() 

# 0.279018+0.227786+0.156094+0.128651+0.101556

# plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative'])
plt.grid()
plt.show()

# Plot PCA and its coefficients
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .12
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attribute_names)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Glass Identification: PCA Component Coefficients')
plt.show()

# Visualize distribution of attributes
fig = figure(figsize=(8,7))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    # i = i + 1
    subplot(u,v,i+1)
    hist(X[attribute_names[i]], color=(0.03*i, 0.8-i*0.05, 0.1*i))
    plt.legend([attribute_names[i]])
    #plt.legend([legendStrs[i]])
    #xlabel(attribute_names[i])
    ylim(0,N/2)
fig.suptitle('Distribution of attributes', size=16)
show()

# Boxplt to check for outliers (normalized)
boxplot(Xt)
xticks(range(1,M+1),attribute_names)
title('Normalized boxplot of Glass Identification')
show()

# Matrix plot (normalized data)
figure(figsize=(12,6))
imshow(Xt, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(M), attribute_names)
xlabel('Attributes')
ylabel('Data objects')
title('Matrix plot of attributtes')
colorbar()
"""

# --Classification--
Xt = Xt.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=42, shuffle=True)

# Multinomial model
regularization_strength = 5
multiLogModel = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-5, max_iter=1000,
                               penalty='l2', C=regularization_strength)
multiLogModel.fit(X_train,y_train)
y_test_est = multiLogModel.predict(X_test)
errorMultiLog = np.sum(y_test_est != y_test) / len(y_test)

# Classification tree model
criterion = 'gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_impurity_decrease=0.01)  
dtc = dtc.fit(X_train, y_train)
y_test_est = dtc.predict(X_test)
errorTreeModel = np.sum(y_test_est != y_test) / len(y_test)

# Baseline model
baseline = np.bincount(y_test).argmax()
errorBaseline = np.sum(baseline != y_test) / len(y_test)


# --Nested cross validation--
outerFolds = 10
innerFolds = 10
cv_outer = KFold(n_splits=outerFolds, shuffle=True, random_state=1)
cv_inner = KFold(n_splits=innerFolds, shuffle=True, random_state=1)

outer_results = np.empty([outerFolds,3])
# yhat = np.empty([N, 3])
yhat = np.empty([1,3])
y_true = np.empty([1])

k=0
for train_index, test_index in cv_outer.split(Xt):
    
    print("Outer split: " + str(k))
    # extract training and test set for current CV fold
    X_train = Xt[train_index,:]
    y_train = y[train_index]
    X_test = Xt[test_index,:]
    y_test = y[test_index]
    
    #fit models
    multiLogModel = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-5, max_iter=1000000,
                                   penalty='l2')
    
    criterion = 'gini'
    decisionTreeModel = tree.DecisionTreeClassifier(criterion=criterion)
    
    baseline = np.bincount(y_train).argmax()
    
    treeSpace = dict()
    treeSpace['min_impurity_decrease'] = np.arange(0, 0.1, 0.001)
    multSpace = dict()
    multSpace['C'] = np.arange(1, 10, 0.1)
    # [1,2,3,4,5,6,7,8,9,10]
    
    treeSearch = GridSearchCV(decisionTreeModel, treeSpace, scoring='accuracy', cv=cv_inner)
    treeResult = treeSearch.fit(X_train, y_train)
    bestTreeModel = treeResult.best_estimator_
    
    multSearch = GridSearchCV(multiLogModel, multSpace, scoring='accuracy', cv=cv_inner)
    multResult = multSearch.fit(X_train, y_train)
    bestMultModel = multResult.best_estimator_
    
    yhatTree = bestTreeModel.predict(X_test).reshape(len(X_test), 1)
    yhatMult = bestMultModel.predict(X_test).reshape(len(X_test), 1)
    baselineValues = np.full((len(X_test), 1), baseline)
    
    dy = np.concatenate((yhatTree, yhatMult, baselineValues), axis=1)
    yhat = np.concatenate((list(np.int_(yhat)), dy), axis=0)
    y_true = np.concatenate((list(np.int_(y_true)), y_test), axis=0)
    
    # dont't ask
    y_test.reset_index(inplace=True, drop=True)
    y_test1 = y_test.to_numpy(copy=True)
    y_test2 = y_test1.reshape(len(y_test), 1)
    
    treeErr = np.sum(yhatTree != y_test2) / len(y_test)
    multErr = np.sum(yhatMult != y_test2) / len(y_test)
    baseErr = np.sum(baseline != y_test2) / len(y_test)
    
    outer_results[k,0]= treeErr
    outer_results[k,1]= multErr
    outer_results[k,2]= baseErr
    
    print("k: " + str(bestTreeModel.min_impurity_decrease))
    print("lambda: " + str(bestMultModel.C))
    k = k+1
    
# Statistical evaluation
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)


    
    
    
    