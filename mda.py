import numpy as np
import matplotlib.pyplot as plt

def maximum_dissimilarity_subset(X, n_subset, l_norm=2):
    # X is a numpy array (instances x features)
    # n_subset is the length of subset to return
    # l_norm is the norm used to evalute distance, default is ecludian l_norm=2
    # Reference: https://www.sciencedirect.com/science/article/abs/pii/S0378383911000354

    # Example
    # X = np.random.rand(100,2)
    # S = maximum_dissimilarity_subset(X, 10)
    
    # First find instance (vector) with greatest dissimilarity, d
    # Preallocate d, with columns are vectors, rows are difference with each vector
    d = np.zeros((N,N)) 
    
    # Calculate difference from each vector (including from itself)
    for i in range(N):
        d[i,:] = np.power(np.sum(np.abs(np.power(X[i,:]-X,l_norm)),1),1/l_norm)
    
    # Sum dissimilarities, find the vector most dissimilar from the others
    d = np.sum(d,axis=1)
    I = np.argmax(d)

    # Move this vector into the subset S
    S = np.expand_dims(X[I,:],axis=1).T
    X = np.delete(X, I, axis=0)

    # Now find the most dissimilar vector from our subset
    # Looping until we have a subset of the desired length 
    for r in range(1,n_subset):
        (Ns,_) = X.shape
        (R,_) = S.shape
        
        # Calculate differences
        d = np.zeros((R,Ns))
        for i in range(R):
            d[i,:] = np.power(np.sum(np.abs(np.power(S[i,:]-X,l_norm)),axis=1),1/l_norm)

        # For each instance take the minimum dissimilarity with the vectors in subset 
        d = np.min(d,axis=0)
        
        # Find the instance with the maximum dissimlarity from above
        I = np.argmax(d)
        
        # Append to subset, remove from data
        S = np.concatenate((S,np.expand_dims(X[I,:],axis=1).T),axis=0)
        X = np.delete(X, I, axis=0)
        
    return S

# 2-dimensional test
X = np.random.rand(100,2)
S = maximum_dissimilarity_subset(X, 10)

# Plot
plt.scatter(X[:,0],X[:,1], label='data')
plt.scatter(S[:,0],S[:,1], label='subset')
plt.legend()