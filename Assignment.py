# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show, boxplot, xticks, ylabel
from matplotlib.pyplot import (imshow, ylabel, title, colorbar, cm)
from scipy.linalg import svd


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

# Compute SVD and variance
U,S,Vh = svd(Xt,full_matrices=False)
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




