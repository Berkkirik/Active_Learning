# Active_Learning
 This algorithm is used for anomali detection

# İmport section
## to run the model , you need to install the packages of python
### To find the packages or more information about libraries of python, u can visit(https://pypi.org/)
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.stats import skew
import featureExtraction
from scipy import signal
import datetime
import pickle
import math


#  LABELS 
## Classes represent the the classes of data
'''

labels = ["fall","walking"] #labels (ex => labels = ["loc1","loc2"])
'''


# DATASET PATH

"""
all_data = "C:\\Users\Berk\Desktop\Bath_Model\Data\_All_data" # Path of Data
'''

# Sample Number of Data
'''

sample_number = 300  #sample number (ex => sample_number = 480)
'''

# Drops and List directory
'''

file_list = os.listdir(all_data)
sample_number = sample_number
labels = labels
d_train=[]
d_trainy=[]
drops = [0,1,2,3,4,5,32,59,60,61,62,63]
'''


# Try-Except Block to detect the Value Error 
'''
for f in file_list:
    tokens = f.split("_")
    train_data=None
    if tokens[3]=="STA1":
        tokens = f.split("_")
        try:
            with open(all_data+"/"+f, 'rb') as f:
                  train_data=np.load(f)
                  
            drops = [0,1,2,3,4,5,32,59,60,61,62,63]
            train_data=np.delete(train_data,drops,1)
            if train_data.shape[0]==sample_number:
                d_trainy.append(labels.index(tokens[0]))
            d_train.append(train_data)
        except ValueError:
            print(f"{f} has 0 size")
            continue
'''

# Train set convert to np.asaray
'''
d_train = np.asarray(d_train,dtype=float)
d_trainy = np.asarray(d_trainy,dtype=float)
'''

# To see the shape of d_train
'''
print("Line 70 df_train_shape", d_train.shape)
'''

# Visualization Section
'''
plt.plot(d_train[0])
plt.show()
plt.show()

'''

# Feature Extraction Method which we import
'''
feature_extraction_method_name = "create_eigens_with_fft" # name of the feature extraciton method (can be:
#feature_extraction_method_name=  "create_eigens"       # "create_eigens", "create_eigens_with_fft", "pca")
feature_extraction_method = getattr(featureExtraction, feature_extraction_method_name)

d_train = feature_extraction_method(d_train)
d_train = np.array(d_train,dtype=float)
values, counts = np.unique(d_trainy, return_counts=True)
 '''


# to fix the data imbalance problem
'''
d_train=d_train[:2*counts[0]]
d_trainy=d_trainy[:2*counts[0]],
'''

# Machine Learning Algorithm Section
from pyod.models.knn import KNN  # kNN detector
#from pyod.models.gmm import GMM  # kNN detector
# train kNN detector
clf_name = 'KNN'
clf = KNN()
clf.fit(d_train)
# get the prediction label and outlier scores of the training data
'''
x_outliers = clf.labels_  # binary labels (0: inliers, 1: outliers)

count=0
x_data=[]
y_data=[]
for i in range(len(d_train)):
    if x_outliers[i]==1:
        count+=1
        continue
    x_data.append(d_train[i])
    y_data.append(d_trainy[i])


x_train, x_dev, y_train, y_dev = train_test_split(x_data, y_data, test_size = 0.2)

x_train=np.asarray(x_train)
x_dev=np.asarray(x_dev)
y_train = np.asarray(y_train) 
y_dev=np.asarray(y_dev)


###Active Learning    
#inital parameters
reg = ExtraTreesClassifier(max_depth=5, n_estimators=20)
epoch=15
k=30
N=50
Dx=x_train.copy()
Dy=y_train.copy()

#intial data sets
Lx=Dx[:N]
Ly=Dy[:N]
Dx=Dx[N:]
Dy=Dy[N:]
dev_test_results=[]



for i in range(epoch):
    #inital model
    reg.fit(Lx, Ly)
    reg.score(Lx, Ly)
    devtest_res=reg.score(x_dev,y_dev)
    print(devtest_res)
    dev_test_results.append(devtest_res)
    print(dev_test_results)
    #probas=reg.predict_proba(Lx)
    
    #enter the loop for remaining data DX
    probas=reg.predict_proba(Dx)
    i=0
    entropi_values=[]
    for i in range(len(probas)):
        data=probas[i]
    #entropi calculations
        s=-1*((data[0]+0.00001)*math.log((data[0]+0.00001),2))-1*((data[1]+0.00001)*math.log((data[1]+0.00001),2))
        #print(i)
        entropi_values.append(s)
    
    
    temp=entropi_values.copy()
    temp=np.sort(temp)
    nlx=[]
    nly=[]
    i=0
    for i in range(1,k+1):
        ind=entropi_values.index(temp[-i])
        #print(ind)
        nlx.append(Dx[ind])
        nly.append(Dy[ind])
    nlx=np.asarray(nlx,dtype=float)
    nly=np.asarray(nly,dtype=float)
    #concatanete preciosu dataset with new selected samples by active learner
    Lx=np.concatenate((Lx,nlx))    
    Ly=np.concatenate((Ly,nly)) 
    
    Dx=Dx[k:]
    Dy=Dy[k:]
    
    #now enter the loop
    
    date_time = str(datetime.datetime.now())
    date_time = date_time.replace(" ",",")
    date_time = date_time.replace(":",".")  
    winSize=300
    step_size=300
    score=dev_test_results[-1]

'''


# Model save section

'''

filename = "activel_new_fall_nonfall_extratre_"+str(winSize)+"_"+str(step_size) +"_"+ str(score)+"_" + date_time[5:10] + ".sav"
pickle.dump(reg, open("./models" + '/' + filename, 'wb'))

'''

# For Prediction of test data
'''
reg.score(x_train,y_train)


'''

# With never seen data , create conf. matrix

from sklearn.metrics import confusion_matrix

y_pred=reg.predict(x_train)
cm=confusion_matrix(y_pred,y_train)
print(cm)
                 
