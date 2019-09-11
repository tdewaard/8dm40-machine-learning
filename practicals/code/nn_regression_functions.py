# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:46:04 2019

@author: s140390

Excercise k-NN Classification - Week 1
"""
#Import all functionalities and the dataset
import numpy as np

def k_NN_Regression(dataset,features,k):
    # Split data in train and test data
    X_train = dataset.data[:300, :]
    y_train = dataset.target[:300, np.newaxis]
    X_test = dataset.data[300:, :]
    y_test = dataset.target[300:, np.newaxis]

    #Compare length for each sample in train dataset and determine nearest neighbour
    results = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        diff = np.zeros(len(X_train))
        diff_orig = []
        for j in range(len(X_train)):
            diff[j] = np.linalg.norm(X_test[i] - X_train[j])
            diff_orig.append(diff[j])
        diff.sort()
            
        #Determine nearest neighbours
        targets=[]
        min_diff=diff[:k]
        for m in range(k):
            index_min=diff_orig.index(min_diff[m])
            targets.append(y_train[index_min])
        
        #Determine class
        estimate = sum(targets)/k
        results[i] = estimate
    
    #Determine MSE
    SE = np.zeros(len(y_test))
    for i in range(len(y_test)):
        SE[i] = np.square(y_test[i] - results[i])
    
    MSE = sum(SE) / len(SE)
    
    return results, MSE, y_test, X_test
            
        
        

    