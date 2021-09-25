#!/usr/bin/env python
# coding: utf-8

# In[1]:


#K-means clustering is powerful to split up dataset into cluster similarity
# In data "1" means YES, "0" means No, and "1.5" for Abstain 


# In[2]:


#import libraries

import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[3]:


#load data using pandas
votes = pd.read_csv("example/114_congress.csv")


# In[4]:


votes.shape #we have 100 senators


# In[5]:


print(votes.shape) # 100 senators and they voted 18 times?


# In[6]:


votes.head() # they voted 15 times, 3 oclumns are descriptive


# In[7]:


votes.tail()


# In[8]:


print(pd.value_counts(votes.iloc[:,3:].values.ravel())) # we have 803 "YES", 669 "No" and abstain "28".


# In[9]:


#initial K-means clustering
# K-means clustering will try to make clusters out of the senators
# Each cluster will contain senators whose votes are as similar to each other as possible
#First specify the number of cluster we want up front. let's try 2 and see how it looks

#The Kmeans algorithm is implemented in the scikits-learn library

from sklearn.cluster import KMeans


# In[10]:


#Create a Kmeans model on our data, using 2 clusters. random_state helps ensure the algorithm returns the same results each time.

kmeans_model = KMeans(n_clusters = 2, random_state = 1).fit(votes.iloc[:, 3:]) #we don't want to include the first 3 columns


# In[11]:


#These are our fitted labels for clusters 

labels = kmeans_model.labels_


#the first cluster has label 0
#the second cluster has label 1


# In[12]:


print(pd.crosstab(labels, votes["party"])) #results according to the parties "D" and "R"

#2 independents are in the democratic cluster, 
#3 democrats in the republcian


# In[13]:


#exploring senators which are in the wrong cluster

#Lets name these types of voters "oddballs"

#There are no republicans in the oddballs

democratic_oddballs = votes[(labels == 1) & (votes["party"] == "D")]


# In[14]:


print(democratic_oddballs["name"])


# In[15]:


democratic_oddballs # It showed that Reid abstained a lot


# In[16]:


democratic_oddballs1 = votes[(labels == 0) & (votes["party"] == "D")]


# In[17]:


print(democratic_oddballs1["name"])


# In[18]:


democratic_oddballs1.head(10)


# In[19]:


#Plotting the cluster using PCA (principal component analysis)
#Each column of data is a dimension on a plot and we have 15 columns which is impossible to visualize all 15 colmn together.
# pCa can help to compress votes columns into two then plot all senators according to their votes then shade them by K-means cluster

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[20]:


pca_2 = PCA(2)


# In[21]:


#Turn the vote data into two columns with PCA
plot_columns = pca_2.fit_transform(votes.iloc[:,3:18])


# In[22]:


votes['label']=[not x for x in labels] # wanting Republicans to be red in scatter plot below


# In[23]:


#Plot senators using PCA (two dimensions) and shade by cluster label

plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = votes["label"])

plt.show()


# In[24]:


plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = votes["label"], s = 100)

plt.show()


# In[25]:


pca_2 = sklearn.decomposition.PCA(2)


# In[26]:


plot_columns = pca_2.fit_transform(votes.iloc[:,3:-1])


# In[27]:


#Plot senators using PCA (two dimensions) and shade by cluster label

plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = votes["label"])

plt.show()


# In[28]:


#More cluster (wings party, cross party group) can show more information about
# Use 5 clustser instead of two clusters

import pandas as pd
from sklearn.cluster import KMeans


# In[29]:


##Create a Kmeans model on our data, using 5 clusters.
kmeans_model = KMeans(n_clusters = 5, random_state = 2).fit(votes.iloc[:, 3:])


# In[30]:


labels = kmeans_model.labels_


# In[31]:


print(pd.crosstab(labels, votes["party"]))


# In[32]:


pca_5 = sklearn.decomposition.PCA(5)


# In[33]:


plot_columns = pca_5.fit_transform(votes.iloc[:,3:-1])


# In[34]:


votes['label']=[not x for x in labels] 


# In[35]:


#Plot senators using PCA (two dimensions) and shade by cluster label

plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = votes["label"])

plt.show()

