import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.utils import shuffle
from scipy.io import arff

import fileinput
from sklearn.model_selection import train_test_split
from sklearn import svm
import glob as gl
seed = 0
seed = 0

"Normalize training and testing sets"
def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X



#**********************************IoT botnet******************************

def IoT_botnet(device_name):
    dataframes_benign = pd.DataFrame()
    dataframes_nonbenign = pd.DataFrame()
    path = '/home/cnm02/DATA/IoT/'+device_name
    all_files = gl.glob(path+'/*.csv')
    for filename in all_files:
        
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe)
            dataframes_benign['class'] = 0
        else:
            dataframes_nonbenign = dataframes_nonbenign.append(dataframe)
            dataframes_nonbenign['class'] = 1
   
    IoT_Botnet_data = dataframes_benign.append(dataframes_nonbenign)
    
    IoT_Botnet_data = shuffle(IoT_Botnet_data)
    num_samples = len (IoT_Botnet_data)
    split = int (0.7*num_samples)
    
    IoT_Botnet_data_train = IoT_Botnet_data[:split]
    IoT_Botnet_data_test = IoT_Botnet_data[split:]
    
   
    
    Y_train = IoT_Botnet_data_train['class']
    Y_train = pd.get_dummies(Y_train)
    
    Y_test = IoT_Botnet_data_test['class']
    Y_test = pd.get_dummies(Y_test)
    
    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test = IoT_Botnet_data_test.drop("class", axis=1)
 
    X_train, X_test = normalize_data(X_train, X_test, 'minmax')
    
    X_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)
    
    return X_train, Y_train, X_test, Y_test, X_0, X_1

#**********************************IoT botnet******************************

def IoT_botnet1(device_name):
    dataframes_benign = pd.DataFrame()
    dataframes_nonbenign_train = pd.DataFrame()
    dataframes_nonbenign_test = pd.DataFrame()
    path = '/home/cnm02/DATA/IoT/'+device_name
    path2 = '/home/cnm02/DATA/IoT/'+device_name+'/test_attack'
    all_files = gl.glob(path+'/*.csv')
    all_files2 = gl.glob(path2+'/*.csv')
    for filename in all_files2:
        dataframe = pd.read_csv(filename)
        dataframes_nonbenign_test = dataframes_nonbenign_test.append(dataframe)
        dataframes_nonbenign_test['class'] = 1
        #Test attack
    for filename in all_files:
        
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe)
            dataframes_benign['class'] = 0
        else:
            dataframes_nonbenign_train = dataframes_nonbenign_train.append(dataframe)
            dataframes_nonbenign_train['class'] = 1
   
    #IoT_Botnet_data = dataframes_benign.append(dataframes_nonbenign)
    
    dataframes_benign = shuffle(dataframes_benign)
    num_samples = len (dataframes_benign)
    split = int (0.1*num_samples)
    
    dataframes_benign_train = dataframes_benign[:split]
    dataframes_benign_test = dataframes_benign[split:]
   
    
    IoT_Botnet_data_train = dataframes_benign_train.append(dataframes_nonbenign_train)
    print ("len train: ", len(IoT_Botnet_data_train))
    IoT_Botnet_data_test = dataframes_benign_test.append(dataframes_nonbenign_test)
    print ("len test: ", len(IoT_Botnet_data_test))
    
    IoT_Botnet_data_train = shuffle (IoT_Botnet_data_train)
    IoT_Botnet_data_test = shuffle (IoT_Botnet_data_test)
    
   
    
    Y_train = IoT_Botnet_data_train['class']
    Y2 = np.asarray(Y_train)
    Y_train = pd.get_dummies(Y_train)
    
    Y_test = IoT_Botnet_data_test['class']
    Y_test = pd.get_dummies(Y_test)
    
    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test = IoT_Botnet_data_test.drop("class", axis=1)
 
    X_train, X_test = normalize_data(X_train, X_test, 'minmax')
    
    X_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)
    X_train, X_test = normalize_data(X_train, X_test, "minmax")
    X_0 = X_train[Y2 == 0]
    X_1 = X_train[Y2 != 0]
   
    
    return X_train, Y_train, X_test, Y_test, X_0, X_1
    
def IoT_botnet2(device_name):
    dataframes_benign = pd.DataFrame()
    dataframes_nonbenign = pd.DataFrame()
    IoT_Botnet_data = pd.DataFrame()
    dataframes_nonbenign_train = pd.DataFrame()
    dataframes_nonbenign_test = pd.DataFrame()
    path = '/home/cnm02/DATA/IoT/'+device_name
    path2 = '/home/cnm02/DATA/IoT/'+device_name+'/test_attack'
    all_files = gl.glob(path+'/*.csv')
    
    for filename in all_files:
        
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe)
            dataframes_benign['class'] = 0
        else:
            dataframes_nonbenign = dataframes_nonbenign.append(dataframe)
            dataframes_nonbenign['class'] = 1
   
    IoT_Botnet_data = dataframes_benign.append(dataframes_nonbenign)
    
    IoT_Botnet_data = shuffle(IoT_Botnet_data)
    num_samples = len (IoT_Botnet_data)
    split = int (0.5*num_samples)
    
    dataframes_benign_train = dataframes_benign[:split]
    dataframes_benign_test = dataframes_benign[split:]

    
    IoT_Botnet_data_train = IoT_Botnet_data [:split] 
    IoT_Botnet_data_test = IoT_Botnet_data [split:] 
    
   
    
    Y_train = IoT_Botnet_data_train['class']
    Y2 = np.asarray(Y_train)
    Y_train = pd.get_dummies(Y_train)
    
    Y_test = IoT_Botnet_data_test['class']
    Y_test = pd.get_dummies(Y_test)
    
    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test = IoT_Botnet_data_test.drop("class", axis=1)
 
    X_train, X_test = normalize_data(X_train, X_test, 'maxabs')
    
    X_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)
    #X_train, X_test = normalize_data(X_train, X_test, "minmax")
    X_0 = X_train[Y2 == 0]
    X_1 = X_train[Y2 != 0]
   
    
    return X_train, Y_train, X_test, Y_test, X_0, X_1
     
    
def IoT_botnet_unknown(device_name):
    dataframes_benign = pd.DataFrame()
    dataframes_gafgyt_combo = pd.DataFrame()
    dataframes_gafgyt_junk = pd.DataFrame()
    dataframes_gafgyt_scan = pd.DataFrame()
    dataframes_gafgyt_tcp = pd.DataFrame()
    dataframes_gafgyt_udp = pd.DataFrame()
    dataframes_mirai_ack = pd.DataFrame()
    dataframes_mirai_scan = pd.DataFrame()
    dataframes_mirai_syn = pd.DataFrame()
    dataframes_mirai_udp = pd.DataFrame()
    dataframes_mirai_udpplain = pd.DataFrame()
    path = '/home/cnm02/DATA/IoT/'+device_name
    all_files = gl.glob(path+'/*.csv')
    for filename in all_files:
        print(filename)
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe)
            dataframes_benign['class'] = 0
        
        elif 'gafgyt_attacks_combo'in filename:
            dataframes_gafgyt_combo = dataframes_gafgyt_combo.append(dataframe)
            dataframes_gafgyt_combo['class'] = 1
        
        elif 'gafgyt_attacks_junk' in filename:
            dataframes_gafgyt_junk = dataframes_gafgyt_junk.append(dataframe)
            dataframes_gafgyt_junk['class'] = 1
            
        elif 'gafgyt_attacks_scan'in filename:
            dataframes_gafgyt_scan = dataframes_gafgyt_scan.append(dataframe)
            dataframes_gafgyt_scan['class'] = 1
            
        elif 'gafgyt_attacks_tcp'in filename:
            dataframes_gafgyt_tcp = dataframes_gafgyt_tcp.append(dataframe)
            dataframes_gafgyt_tcp['class'] = 1
           
            
        elif 'gafgyt_attacks_udp'in filename:
            dataframes_gafgyt_udp = dataframes_gafgyt_udp.append(dataframe)
            dataframes_gafgyt_udp['class'] = 1
            
        elif 'mirai_attacks_ack'in filename:
            dataframes_mirai_ack = dataframes_mirai_ack.append(dataframe)
            dataframes_mirai_ack['class'] = 2
        
        elif 'mirai_attacks_scan'in filename:
            dataframes_mirai_scan = dataframes_mirai_scan.append(dataframe)
            dataframes_mirai_scan['class'] = 2
    
        elif 'mirai_attacks_syn'in filename:
            dataframes_mirai_syn = dataframes_mirai_syn.append(dataframe)
            dataframes_mirai_syn['class'] = 2
        
        elif 'mirai_attacks_udp'in filename:
            dataframes_mirai_udp = dataframes_mirai_udp.append(dataframe)
            dataframes_mirai_udp['class'] = 2
        
        elif 'mirai_attacks_udpplain' in filename:
            dataframes_mirai_udpplain = dataframes_mirai_udpplain.append(dataframe)
            dataframes_mirai_udpplain['class'] = 2

    IoT_Botnet_data = dataframes_benign.append(dataframes_gafgyt_combo)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_junk)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_scan)
    #dataframes_gafgyt_tcp = shuffle(dataframes_gafgyt_tcp)
    #dataframes_gafgyt_tcp = dataframes_gafgyt_tcp[0:int(0.5*len(dataframes_gafgyt_tcp))]
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_tcp)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_udp)
    #IoT_Botnet_data_unknown = dataframes_gafgyt_udp
    #IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_ack, sort= True)
    #IoT_Botnet_data_unknown = dataframes_mirai_scan
    #IoT_Botnet_data_unknown = dataframes_mirai_ack 
    IoT_Botnet_data_unknown = dataframes_mirai_ack
    IoT_Botnet_data_unknown = IoT_Botnet_data_unknown.append(dataframes_mirai_scan)
    IoT_Botnet_data_unknown = IoT_Botnet_data_unknown.append(dataframes_mirai_syn)
    IoT_Botnet_data_unknown = IoT_Botnet_data_unknown.append(dataframes_mirai_udp)# dataframes_gafgyt_udp.append(dataframes_mirai_udp)
    IoT_Botnet_data_unknown = IoT_Botnet_data_unknown.append(dataframes_mirai_udpplain)
    IoT_Botnet_data_unknown = shuffle(IoT_Botnet_data_unknown)
    #IoT_Botnet_data_unknown = IoT_Botnet_data_unknown [0:int (0.05*len(IoT_Botnet_data_unknown))]
    
    
    IoT_Botnet_data = shuffle(IoT_Botnet_data)
    l = len (IoT_Botnet_data)
    IoT_Botnet_data_train = IoT_Botnet_data[:int(0.8*l)]
    IoT_Botnet_data_test = IoT_Botnet_data[int(0.8*l):]
    IoT_Botnet_data_test_3class = IoT_Botnet_data_test.append(IoT_Botnet_data_unknown)
    IoT_Botnet_data_test_3class = shuffle(IoT_Botnet_data_test_3class)
    
    dataframes_gafgyt_udp['class'] = 1
    dataframes_mirai_udp['class'] = 1
    dataframes_mirai_scan['class'] = 1
    IoT_Botnet_data_unknown['class'] = 1
    
    IoT_Botnet_data_unknown =dataframes_mirai_udp# dataframes_gafgyt_udp.append(dataframes_mirai_udp)
    IoT_Botnet_data_test_2class = IoT_Botnet_data_test.append(IoT_Botnet_data_unknown)
    IoT_Botnet_data_test_2class = shuffle(IoT_Botnet_data_test_2class)
    
    
    Y_train = IoT_Botnet_data_train['class']
    Y_tr = np.array(Y_train)
   
    unique2, counts2 = np.unique(Y_tr, return_counts=True)
    print (dict(zip(unique2, counts2)))
    Y_train = pd.get_dummies(Y_train)
    
    Y_test_3 = IoT_Botnet_data_test_3class['class']
    Y_tr = np.array(Y_test_3)
    print (Y_tr[0:10])
    unique2, counts2 = np.unique(Y_tr, return_counts=True)
    print (dict(zip(unique2, counts2)))
    Y_test_3 = pd.get_dummies(Y_test_3)
    print (Y_test_3[0:10])

    
    Y_test_2 = IoT_Botnet_data_test_2class['class']
    Y_test_2 = pd.get_dummies(Y_test_2)
    
    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test_3 = IoT_Botnet_data_test_3class.drop("class", axis=1)
    X_test_2 = IoT_Botnet_data_test_2class.drop("class", axis=1)
    
    X_train_2, X_test_2 = normalize_data(X_train, X_test_2, 'maxabs')
    X_train_3, X_test_3 = normalize_data(X_train, X_test_3, 'maxabs')
    
   
    #X_10 = X_train[IoT_Botnet_data_train["class"] == 10]
    
    X_train_2 = np.array(X_train_2)
    X_train_3 = np.array(X_train_3)
    Y_train_2 = np.array(Y_train)
    Y_train_3 = np.array(Y_train)
    X_test_2 = np.array(X_test_2)
    X_test_3 = np.array(X_test_3)
    Y_test_2 = np.array(Y_test_2)
    Y_test_3 = np.array(Y_test_3)

    return X_train_2, Y_train_2, X_test_2, Y_test_2, X_train_3, Y_train_3, X_test_3, Y_test_3
    
"*****************************Load dataset*****************************"
def read_data_sets(dataset):
    path_data = "/home/cnm02/DATA/"
    
    if (dataset == "Danmini_Doorbell"):
        #X_train, Y_train, X_test, Y_test, X0, X1 =  IoT_botnet1("Danmini_Doorbell")
        #test for unknown
        X_train_2, Y_train_2, X_test_2, Y_test_2, X_train_3, Y_train_3, X_test_3, Y_test_3 = IoT_botnet_unknown ("Danmini_Doorbell") 
    elif (dataset == "Ecobee_Thermostat"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Ecobee_Thermostat")
    elif (dataset == "Ennio_Doorbell"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Ennio_Doorbell")
    elif (dataset == "Philips_B120N10_Baby_Monitor"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Philips_B120N10_Baby_Monitor")
    elif (dataset == "Provision_PT_737E_Security_Camera"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Provision_PT_737E_Security_Camera")
    elif (dataset == "Provision_PT_838_Security_Camera"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Provision_PT_838_Security_Camera")
    elif (dataset == "Samsung_SNH_1011_N_Webcam"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("Samsung_SNH_1011_N_Webcam")    
    elif (dataset == "SimpleHome_XCS7_1002_WHT_Security_Camera"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("SimpleHome_XCS7_1002_WHT_Security_Camera")
    elif (dataset == "SimpleHome_XCS7_1003_WHT_Security_Camera"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet1("SimpleHome_XCS7_1003_WHT_Security_Camera")
    elif (dataset == "Ads"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Internet_ads()
    elif (dataset == "s_"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet2("Ecobee_Thermostat")
    elif (dataset == "target"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet2("Ennio_Doorbell")
    else:
        print ("Incorrect data")

    class DataSets(object):
          pass
    data_sets = DataSets()
    #data_sets.train = DataSet(X_train, Y_train)
    #data_sets.test = DataSet(X_test, Y_test)
    #For unknown attack
    data_sets.train = DataSet(X_train_3, Y_train_3)
    data_sets.test = DataSet(X_test_3, Y_test_3)

    return data_sets

class DataSet(object):
  def __init__(self, features, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert features.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (features.shape,
                                                 labels.shape))
    self._num_examples = features.shape[0]
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
   
  @property
  def features(self):
    return self._features
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed

  
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]

 

