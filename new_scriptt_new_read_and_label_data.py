# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:15:08 2023

@author: Berk
"""
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




#classes
labels = ["fall","walking"] #labels (ex => labels = ["loc1","loc2"])

#dataset path
#folder_name = "data/new_data_labeling/fall_nonfall"

all_data = "C:\\Users\Berk\Desktop\Bath_Model\Data\_All_data" # Path of Data

all_data = "C:\\Users\Berk\Desktop\Bath_Model\Data\_NS_data" # path of never seen data

sample_number = 300  #sample number (ex => sample_number = 480)

#"""
file_list = os.listdir(all_data)
sample_number = sample_number
labels = labels
d_train=[]
d_trainy=[]
drops = [0,1,2,3,4,5,32,59,60,61,62,63]
#"""


#"""
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


d_train = np.asarray(d_train,dtype=float)
d_trainy = np.asarray(d_trainy,dtype=float)

print("Line 70 df_train_shape", d_train.shape)
#"""




"""
wden = getattr(denoise, 'wden')   

denoised_d_train = np.empty(d_train.shape)
for n in range(d_train.shape[0]):
    for c in range(d_train.shape[2]):
        channel = d_train[n][:, c]
        #dwd2_temp = pywt.dwt(data=channel, wavelet='sym6', mode='symmetric')
        denoised_temp = wden(channel, tptr = 'heursure', sorh = 'soft', scal = 'one', n = 2, wname = 'sym6' )
        denoised_d_train[n][:, c] = denoised_temp      
        #break
    #break
#print("denoised_temp.shape", denoised_temp.shape)

#"""


plt.plot(d_train[0])
plt.show()
# plt.plot(denoised_d_train[0])
plt.show()



# pca = PCA()
# pc_d_train = np.empty(denoised_d_train.shape)
# for i in range(denoised_d_train.shape[0]):
#     pc_temp = pca.fit_transform(denoised_d_train[i])
#     pc_d_train[i] = pc_temp
    
    
# plt.plot(pc_d_train[0])
# plt.show()





# # POWER SPECTRAL DENSITY
# #PSD_d_train = np.array(PSD_d_train)
# from scipy.signal import welch
# psd_freqs, psd_d_train = welch(pc_d_train)
# haart_d_train = []
# for i in range(psd_d_train.shape[0]):
#     temp_haart = pywt.wavedec2(psd_d_train[i], 'haar', level=5)
#     haart_d_train.append(temp_haart)
# haart_d_train = np.array(haart_d_train)

# plt.title("haart_d_train Detal-5 index-1")
# plt.plot(haart_d_train[0][5][1])
# plt.show()





# psd_d_train2, psd_energy = [], []
# for i in range(1125):
#     temp_psd, energy = compute_psd(pc_d_train[i])
#     psd_d_train2.append(temp_psd)
#     psd_energy.append(energy)
# psd_d_train2, psd_energy = np.array(psd_d_train2), np.array(psd_energy)
    
# haart_d_train2 = []
# for i in range(psd_d_train2.shape[0]):
#     temp_haart2 = pywt.wavedec2(psd_d_train2[i], 'haar', level=5)
#     haart_d_train2.append(temp_haart2)
# haart_d_train2 = np.array(haart_d_train2)

# plt.title("haart_d_train2 Detal-5 index-1")
# plt.plot(haart_d_train2[0][5][1])
# plt.show()



# plt.title("psd2 for first sample")
# plt.plot(psd_d_train2[0])
# plt.show()
# plt.title("energy for all samples")
# plt.plot(psd_energy)
# plt.show()


# #CENTER OF ENERGY
# ce_d_train = np.empty((pc_d_train.shape[0], pc_d_train.shape[2]))
# for i in range(pc_d_train.shape[0]):
#     for c in range(pc_d_train.shape[2]):
#         temp_ce = center_of_energy(pc_d_train[i][c])
#         ce_d_train[i][c] = temp_ce

        
#  plt.title("c_energy for sample 50")
#  plt.plot(ce_d_train[50])
#  plt.show()  
#  plt.title("c_energy for sample -50")
#  plt.plot(ce_d_train[-50])
#  plt.show()  


# # TRAINABLE FEATURE EXTRATION
# # FROM PSD
# stats_d_train = np.empty((pc_d_train.shape[0],pc_d_train.shape[1], 6))
# for sample in range(stats_d_train.shape[0]):
#     for channel in range(stats_d_train.shape[1]):
#         temp_channel = pc_d_train[sample][:, channel].reshape((-1,1))
#         print(temp_channel.shape)
#         break
#     break
        




# #convert y to categorical
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
# enc = enc.fit(train_labels)
# train_labels = enc.transform(train_labels)

# input_shape = PCS_train[0].shape

# train_data, train_labels = shuffle(PCS_train, train_labels, random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.9, random_state=1)


















# d_train_pca=d_train[-5]
# pca = PCA()
# pca.fit(d_train_pca)
# d_train_pca = pca.transform(d_train_pca)


# transformed1 = np.abs(np.fft.fft(d_train_pca[:,5])) / 100



feature_extraction_method_name = "create_eigens_with_fft" # name of the feature extraciton method (can be:
#feature_extraction_method_name=  "create_eigens"       # "create_eigens", "create_eigens_with_fft", "pca")
# feature_extraction_method_name = "create_eigens" 
# feature_extraction_method_name = "create_fft" 
feature_extraction_method = getattr(featureExtraction, feature_extraction_method_name)
 

d_train = feature_extraction_method(d_train)

d_train = np.array(d_train,dtype=float)

#print("Line 98, df_train.shape", d_train.shape)

values, counts = np.unique(d_trainy, return_counts=True)

# print("Line 102 counts", counts)
# print("Line 103 values", values)

#for imbalance problem
# d_train=d_train[:2*counts[0]]
# d_trainy=d_trainy[:2*counts[0]]



from pyod.models.knn import KNN  # kNN detector
#from pyod.models.gmm import GMM  # kNN detector
# train kNN detector
clf_name = 'KNN'
clf = KNN()
clf.fit(d_train)
# get the prediction label and outlier scores of the training data
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



# from sklearn.feature_selection import SelectKBest, chi2
# import numpy as np
# import pandas as pd

# # Assuming x_train is your feature matrix with shape (n_samples, 14) and y_train is the target variable
# x_train_df = pd.DataFrame(x_train, columns=["feature_"+str(i) for i in range(11)])
# y_train_df = pd.DataFrame(y_train, columns=["target"])

# # Select the k best features using chi2 test
# selector = SelectKBest(chi2, k=11)
# selector.fit(x_train_df, y_train_df)
# mask = selector.get_support()

# # Extract the selected features using the boolean mask
# selected_features = x_train_df.columns[mask]

# # Transform the data to keep only the selected features
# X_new = selector.transform(x_train_df)

# # Train your classifier using the reduced feature matrix
# clf.fit(X_new, y_train_df)






filename = "activel_new_fall_nonfall_extratre_"+str(winSize)+"_"+str(step_size) +"_"+ str(score)+"_" + date_time[5:10] + ".sav"
pickle.dump(reg, open("./models" + '/' + filename, 'wb'))


filename = "activel_new_nobody_extratre_"+str(winSize)+"_"+str(step_size) +"_"+ str(score)+"_" + date_time[5:10] + ".sav"
pickle.dump(reg, open("./models" + '/' + filename, 'wb'))



reg.score(x_train,y_train)




from sklearn.metrics import confusion_matrix











y_pred=reg.predict(x_train)
cm=confusion_matrix(y_pred,y_train)
print(cm)
                 


