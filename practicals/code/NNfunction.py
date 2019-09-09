def NNfunction(NN,dataset):
    import numpy as np
    """
    Appoint the dataset to a variable value and devide the dataset into a training and testing subdataset.
    """
    Z=dataset.data
    Z_train=Z[:300]
    Z_test=Z[300:]
    """
    Appoint the targets( whether the given datasamples are malignant or healthy) and store this information in the variables : 
    target_train and target_test.
    """
    target_train = dataset.target[:300, np.newaxis]
    target_test= dataset.target[300:, np.newaxis]
    """
    Create a matrix where the normalized dataset can be stored, and after that normalize
    the test and train data.
    """
    norm_z_train=np.zeros(Z_train.shape)
    norm_z_test=np.zeros(Z_test.shape)
    for k in range(30):
        absoluut=np.linalg.norm(Z_train[:,k])
        normalized=Z_train[:,k]/absoluut
        norm_z_train[:,k]=normalized

    for k in range(30):
        absoluut=np.linalg.norm(Z_test[:,k])
        normalized=Z_test[:,k]/absoluut
        norm_z_test[:,k]=normalized
    """
    The classification results are stored in the variable Results. In the for loop, the distances between the sample from the 
    test data, and all of the training data is calculated and stored in the the list distance. After sorting this list, the k nearest 
    neighbours ( with minimal distance to the sample) were evaluated and their targets were averaged and the sample was given the value 0 or 1.
    """
    Results=np.zeros((len(target_train),1))
    
    for i in range(len(target_test)):
        distance=[]
        originaldist=[]
        targets=[]
        for j in range(len(target_train)):
            verschil=abs(np.linalg.norm(norm_z_test[i,:])-np.linalg.norm(norm_z_train[j,:]))
            distance.append(verschil)
            originaldist.append(verschil)
        distance.sort()
        minimaldistance=distance[:NN]
        for k in range(len(minimaldistance)):
            index_min=originaldist.index(minimaldistance[k])
            targets.append(dataset.target[index_min])
        clas=sum(targets)/NN
        if clas>=0.5:
            Results[i,:]=1
        else:
            if  clas<0.5:
                Results[i,:]=0
    return Results