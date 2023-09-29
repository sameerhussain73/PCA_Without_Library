#!/usr/bin/env python
# coding: utf-8

# # Data Mining Homework 1

# By Sameer Hussain

# In[103]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm


# In[113]:


df=pd.read_csv("C:/Users/13177/Downloads/magic04.data",header=None)
df.columns = ['flength','fwidth','fsize','fconc','fconc1','fasym','fm3long','fm3trans','falpha','fdist','class']


# In[114]:


class_1 = df['class']


# In[115]:


df=df.drop(columns=['class'])


# In[127]:


df


# # Question A

# Write a script to apply z-normalization to this dataset. For the reamining questions, you should use z-normalized dataset instead of the original dataset.

# In[14]:


df1=df.to_numpy()
standard=df1.copy()


# In[17]:


standard = (standard-np.mean(standard,axis = 0))/np.std(standard,axis =0)


# In[18]:


standard


# # Question B

# Compute the sample covariance matrix (Σ) as sum of outer product of each
# centered point (see Equation 2.31). Verify that your answer matches with the one using
# numpy.cov function. For both the cases normalize by n instead of n −1

# In[24]:


def out_prod_s(a):
  i=0
  s=0
  while i<=len(a)-1:
    s=s+np.dot(a[i].reshape(a.shape[1],1),a[i].reshape(1,a.shape[1]))
    i=i+1
  return(s/(len(a)-1))


# In[25]:


cov = out_prod_s(standard)
cov


# To cross check with np.cov()

# In[27]:


cov_mat = np.cov(standard,rowvar = False)
cov_mat


# # Question C

# Compute the dominant eigenvalue and eigenvector of the covariance matrix Σ
# via the power-iteration method.Verify your answer using the numpy linalg.eig function.

# In[28]:


def relativeError ( x_n , x_o ) :
    return abs ( ( x_n - x_o ) / x_n ) * 100


# In[37]:


x_diff=0
r = pd.DataFrame ( columns = [ ' eigenvalue ' , ' error ' ] )
x = np.ones ( [ 10,10 ] )
max_iterations=9
OldValue = 1
for i in range (9) :
   x = np.dot ( cov_mat , x )
   eigenval = np.linalg.norm ( x )
   error = relativeError ( eigenval , OldValue )
   x = x / eigenvalue
   r.loc[i]= [eigenvalue, error]
   oldEigenvalue = eigenvalue
   while np.linalg.norm(x_diff)>0.000001:
        x_diff=x[len(x)-1]-x[len(x)-2]

    
r


# To cross check with np.linalg.eig()

# In[36]:


EigenValue, EigenVector = np.linalg.eig(cov_mat)


# In[38]:


EigenValue, EigenVector


# # Question D

# Use linalg.eig to find the first two dominant eigenvectors of Σ, and compute the projection
# of data points on the subspace spanned by these two eigenvectors. Now, compute the variance
# of the datapoints in the projected subspace

# In[52]:


EigenValue = np.sort(EigenValue)[::-1]
EigenVector = np.sort(EigenVector)[::-1]


# In[53]:


EigenVector


# In[54]:


Vec_1 = EigenVector[:,0]
Vec_2 = EigenVector[:,1]


# In[55]:


Vec_1


# In[78]:


Vec_2


# In[57]:


U = np.dstack((Vec_1, Vec_2)).reshape(10,2)
U.shape


# In[60]:


proj_data = np.dot(standard, U)
proj_data


# In[61]:


proj_data.var()


# The variance of the projected data is 2.4422

# # Question E

# Use linalg.eig to find all the eigenvectors, and print the covariance matrix Σ
# in its eigen-decomposition form (UΛUT)

# In[66]:


lambda_1 = np.diag(EigenValue)
lambda_1


# In[64]:


eigen_decomp = np.dot(EigenVector,np.dot(lambda_1,EigenVector.transpose()))


# In[65]:


eigen_decomp


# # Question F

# Compute MSE value when the data points are projected on the space spanned by the first two eigenvectors of the covariance matrix and show that it equals to the sum of the eigenvalues expect the first two.

# In[67]:


Val_1 = EigenValue[0]
Val_2 = EigenValue[1]


# In[72]:


mean_squared_error = np.trace(cov_mat) - (Val_1 + Val_2)
mean_squared_error


# In[75]:


residual_eigen = EigenValue.sum() -(EigenValue[0]+EigenValue[1])
residual_eigen


# In[76]:


mean_squared_error - residual_eigen


# The diffrence is negligible

# # Question G

# Use the class label which we ignored in the other questions. Plot the datapoints after projecting those in the first two principal components.
# Use different colors for instances of different classes.

# In[106]:


def PCA(X , num_components):
     
    #Mean
    X_meaned = X - np.mean(X , axis = 0)
     
    #Covariance
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Eigen value and vector
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


# In[107]:


#prepare the data
x = df.iloc[:,0:10]
 
#prepare the target
target = df.iloc[:,10]
 
#Applying it to PCA function
mat_reduced = PCA(x , 7)
 
#Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2','3','4','5','6','7'])
 
#Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)


import seaborn as sb
import matplotlib.pyplot as plt
 
plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , hue = 'class' , s = 60 , palette= 'icefire')


# # Question H

# Write a subroutine to implement PCA Algorithm. Use the program
# above and find the principle vectors that we need to preserve 95% of variance? Print the co-ordinate of the first 10 data points by using the above set of vectors as the new basis vector.

# Using PCA function from sklearn

# In[108]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
dim_red = pca.fit_transform(standard)


# In[109]:


pd.DataFrame(dim_red)


# PCA Algorithm

# Step 1 - Mean

# In[117]:


mu = df.mean(axis = 0)


# Step 2 - Centered Data

# In[118]:


centered = (df-np.mean(df,axis = 0))/np.std(df,axis =0)


# Step 3 - Covariance Matrix

# In[119]:


covariance_matrix = np.cov(centered, rowvar= False)


# Step 4 - Eigen Values and Vectors

# In[120]:


e_val, e_vec = np.linalg.eig(covariance_matrix)


# Step 5 - r selection

# Best value of r is 7

# Step 6 - Basis Vector

# In[121]:


Ur = e_vec[0:7]


# Step 7 - Reduced Dimension Data

# In[122]:


project_d = centered.dot(e_vec)


# In[123]:


Ai = Ur.dot(project_d.transpose())
Ai.transpose().shape


# In[124]:


Ai


# The shape of the reduced data is (19020,7)
