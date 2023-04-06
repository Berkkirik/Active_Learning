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