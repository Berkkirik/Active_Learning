import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime
import pickle


def create_eigens(d_samples):
    d_train = []
    for i in range(len(d_samples)):
        temp = d_samples[i]
        temp = temp.astype(float)
        corcmatrix_A = np.corrcoef(temp,rowvar=False)
        corcmatrix_A = np.nan_to_num(corcmatrix_A)
        w, v = np.linalg.eig(corcmatrix_A)
        eigens = np.array([w[0],w[1],w[2],w[3],w[4]])
        d_train.append(eigens)
    return d_train
    
   
    
   
def create_fft(d_samples):
    d_train = []
    for i in range(len(d_samples)):
        temp = d_samples[i]
        temp = temp.astype(float)
        # corcmatrix_A = np.corrcoef(temp,rowvar=False)
        # corcmatrix_A = np.nan_to_num(corcmatrix_A)
        # w, v = np.linalg.eig(corcmatrix_A)
        
        sample_signal1 = temp[:,0]
        sample_signal2 = temp[:,10]
        sample_signal3 = temp[:,20]
        sample_signal4 = temp[:,30]
        sample_signal5 = temp[:,40]
        sample_signal6 = temp[:,50]
        # data=temp[:,:6]
        # analytic = continuous.get_h_mvn(data)
        transformed1 = np.abs(np.fft.fft(sample_signal1)) / 100
        # mn1=transformed1.mean()
        transformed2 = np.abs(np.fft.fft(sample_signal2)) / 100
        # mn2=transformed2.mean()
        transformed3 = np.abs(np.fft.fft(sample_signal3)) / 100
        # mn3=transformed3.mean()
        transformed4 = np.abs(np.fft.fft(sample_signal4)) / 100
        # mn4=transformed4.mean()
        transformed5 = np.abs(np.fft.fft(sample_signal5)) / 100
        # mn5=transformed5.mean()
        transformed6 = np.abs(np.fft.fft(sample_signal6)) / 100
        # mn6=transformed6.mean()
        sorted_t1=np.sort(transformed1)
        sorted_t2=np.sort(transformed2)
        sorted_t3=np.sort(transformed3)
        eigens = np.array([sorted_t1[-2],sorted_t1[-4],sorted_t1[-6],
                           sorted_t2[-2],sorted_t2[-4],sorted_t2[-6]])
        # eigens=np.array([w[1],w[2],w[3])
        d_train.append(eigens)
    return d_train 
        
def create_eigens_with_fft(d_samples):
    d_train = []
    for i in range(len(d_samples)):
        temp = d_samples[i]
        temp = temp.astype(float)
        corcmatrix_A = np.corrcoef(temp,rowvar=False)
        corcmatrix_A = np.nan_to_num(corcmatrix_A)
        w, v = np.linalg.eig(corcmatrix_A)
        
        sample_signal1 = temp[:,0]
        sample_signal2 = temp[:,10]
        sample_signal3 = temp[:,20]
        sample_signal4 = temp[:,30]
        sample_signal5 = temp[:,40]
        sample_signal6 = temp[:,50]
        # data=temp[:,:6]
        # analytic = continuous.get_h_mvn(data)
        transformed1 = np.abs(np.fft.fft(sample_signal1)) / 100
        # mn1=transformed1.mean()
        transformed2 = np.abs(np.fft.fft(sample_signal2)) / 100
        # mn2=transformed2.mean()
        transformed3 = np.abs(np.fft.fft(sample_signal3)) / 100
        # mn3=transformed3.mean()
        transformed4 = np.abs(np.fft.fft(sample_signal4)) / 100
        # mn4=transformed4.mean()
        transformed5 = np.abs(np.fft.fft(sample_signal5)) / 100
        # mn5=transformed5.mean()
        transformed6 = np.abs(np.fft.fft(sample_signal6)) / 100
        # mn6=transformed6.mean()
        sorted_t1=np.sort(transformed1)
        sorted_t2=np.sort(transformed2)
        sorted_t3=np.sort(transformed3)
        sorted_t4=np.sort(transformed4)
        sorted_t5=np.sort(transformed5)
        
        eigens = np.array([w[0],w[1],w[2],w[3],w[4],w[5],sorted_t1[-1],sorted_t2[-2],sorted_t3[-1],sorted_t4[-2],sorted_t5[-3]])
        # eigens=np.array([w[1],w[2],w[3])
        d_train.append(eigens)
    return d_train 
    

# PRiNCİPAL COMPONENT ANALYSİS
def pca(d_samples):
    d_samples_subs1=d_samples[:,:,:3]
    d_samples_subs2=d_samples[:,:,23:26]
    d_samples_subs3=d_samples[:,:,49:]

    d_samples_subs = np.concatenate((d_samples_subs1,d_samples_subs2,d_samples_subs3), axis=2)
    
    d_samples = d_samples_subs.reshape(d_samples_subs.shape[0],-1)
    #scaler = StandardScaler()
    #scaler.fit(d_samples)
    #scaled_data = scaler.transform(d_samples)
    pca = PCA()
    pca.fit(d_samples)
    d_train = pca.transform(d_samples)
    d_train_sorted = np.sort(-d_train, axis=1)
    d_train_sorted = -d_train_sorted
    
    index = -150
    
    d_train_con = d_train_sorted[:,index:]
    
    date = str(datetime.datetime.now().date())
    date_time = str(datetime.datetime.now())
    date_time = date_time.replace(" ",",")
    date_time = date_time.replace(":",".")

    path = "models" + "/" + "model_" + date
   
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "pca_" + date_time + ".pkl"
    pickle.dump(pca, open(path + '/' + filename, 'wb'))
    
    return d_train_con














