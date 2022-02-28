# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

file_path = 'Data'

# Gets attribute names
attribute_names = []
with open(file_path+'\glass.names', "r") as filestream:
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
with open(file_path+'\glass.names', "r") as filestream:
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
X = pd.read_csv(file_path+'\glass.data', sep=',', names=attribute_names)

# Clean up data
y = X['Type of glass']
X.drop('Type of glass', inplace=True, axis=1)
X.drop('Id number', inplace=True, axis=1)

