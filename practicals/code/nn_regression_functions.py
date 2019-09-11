# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:46:04 2019

@author: s140390

Excercise k-NN Classification - Week 1
"""
#Import all functionalities and the dataset
import numpy as np

def k_NN_Regression(dataset,k):
    
    """
    dataset = The combination of training and test datasets
    k = The desired amount of neighbours used to estimate the value
    
    Appoint the dataset to a variable value and devide the dataset into a training and testing subdataset.
    """
    
    # Split data in train and test data
    X_train = dataset.data[:300, :]
    y_train = dataset.target[:300, np.newaxis]
    X_test = dataset.data[300:, :]
    y_test = dataset.target[300:, np.newaxis]

    
    """
    The results are stored in the variable 'results'. In the for loop, the distances between the sample from the 
    test data, and all of the training data is calculated and stored in the the list diff. After sorting this list, the k nearest 
    neighbours (with minimal distance to the sample) were evaluated and the corresponding targets were used to estimate the test value.
    """
    
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
    
    
    """
      The Mean Squared Error (MSE) is calculated to evaluate the model. The MSE is defined by the difference between the 
      original target value and the predicted target value, squared. 
    """
    
    #Determine MSE
    SE = np.zeros(len(y_test))
    for i in range(len(y_test)):
        SE[i] = np.square(y_test[i] - results[i])
    
    MSE = sum(SE) / len(SE)
    
    return results, MSE
            
        
        

    