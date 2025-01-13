import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = pd.read_csv('545_cluster_dataset programming 3.txt',sep = '\s+', header= None)
file.columns = ['x','y']
files = file.to_numpy()  ##converting to numpy 
m = 2  ## fuxxifier parameter
cluster_num =[3,6,10]   # different clusters
runs = [20,40,80]   # different runs
for k in cluster_num:
    sse = []
    for r in runs:
        centroid_mem = np.random.random_sample((len(files),k))   ## initially assign membership grades
        sum_mem = centroid_mem / centroid_mem.sum(1,keepdims= True)  ## making sure the sum of each row equals 1
        for _ in range(r):
            cent = np.zeros((k,2))  
            for i in range(k):
                cent[i,:] = np.sum(np.expand_dims(sum_mem[:,i] ** m, axis= 1) * files,axis=0) / np.sum(sum_mem[:,i] ** m) ### calculating centroid

            tance =  np.zeros((len(files),k)) 
            for i in range(k):
                tance[:,i] = np.linalg.norm(files - cent[i,:], axis = 1) 
            update = 1 / (tance**(2/(m -1)) * np.sum((1/tance)**(2/(m-1)), axis=1)[:,np.newaxis])  ## calculating membership

            if np.linalg.norm(update - sum_mem) <= 0.0000005:
                break
            sum_mem = update
            t = np.argmax(update,axis=1)  ##maximum label

             ## computing sum square error
            s = 0
        for i in range(len(files)):    
            ss = np.sum(np.linalg.norm(files[i] - cent[t[i]]) , axis =0)
            s +=ss
        sse.append(ss)
        ##### visualization ###
        sns.scatterplot(x = files[:,0], y = files[:,1],hue = t, palette='bright')
        plt.title(f'FUZZY C MEANS: A total of {r} runs for {k} clusters')
        plt.show()
  ## printing sse and minimum sse across all the runs
    print(sse)
    print(np.min(sse))
    
   
            
