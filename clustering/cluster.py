import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = pd.read_csv('545_cluster_dataset programming 3.txt',sep = '\s+', header= None)
file.columns = ['x','y']
files = file.to_numpy()   ##converting to numpy
cluster_num = [3,5,7]   # different clusters
runs = [10,50,100]   # different runs
for k in cluster_num:
    sse = []
    for r in runs:
        minute = np.zeros(files.shape[0])  # initializing the minimum distance between data points and centroids
        centroid = files[np.random.choice(files.shape[0],k,replace=False)]  # randomly initializing cluster centroid
        
        for _ in range(r):
            old = minute.copy()
            cluster_centroid = np.linalg.norm(np.reshape(files,(1500,1,2)) - np.reshape(centroid,(1,k,2)),axis=2, ord=2) # taking the euclidean distance
            minute = np.argmin(cluster_centroid, axis=1)  # obtaining the index of the minimum distance
            for i in range(k):
                centroid[i,:] = np.mean(files[minute == i], axis= 0)  # computing the mean in order to reassign new centroids
            if all(minute == old):
                break
        ## computing sum square error
        s = 0
        for i in range(len(files)):    
            ss = np.sum(np.linalg.norm(files[i]- centroid[minute[i]]), axis =0)   # sum of distance between datapoints ans assigned centroid
            s +=ss
        sse.append(ss)
        ##### visualization ###
        sns.scatterplot(x = files[:,0], y = files[:,1], data = file, hue = minute, palette='bright')  
        plt.title(f'K MEANS : A total of {r} runs for {k} clusters')
        plt.show()

   ##### printing sse and minimum sse across all the runs
    print(sse)  
    print(np.min(sse)) 

        