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

def IoT_botnet():
    dataframes_benign = pd.DataFrame()
    dataframes_nonbenign_train = pd.DataFrame()
    dataframes_nonbenign_test = pd.DataFrame()
    path = '/home/vuly/DATA/IoT'
    all_files = gl.glob(path+'/*.csv')
    for filename in all_files:
     
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe)
            dataframes_benign['class'] = 0
        else:
            dataframes_nonbenign_train = dataframes_nonbenign_train.append(dataframe)
            dataframes_nonbenign_train['class'] = 1
   
    dataframes_benign = shuffle(dataframes_benign)
    l = int (0.7 * len (dataframes_benign))
    IoT_Botnet_benign_train = dataframes_benign[:l]
    IoT_Botnet_benign_test = dataframes_benign[l:]
    
    IoT_Botnet_data_train = IoT_Botnet_benign_train.append(dataframes_nonbenign_train)
    
    #IoT_Botnet_data_train = shuffle(IoT_Botnet_data_train)
    
    #dataframes_nonbenign_train = shuffle(dataframes_nonbenign_train)
    
    #dataframes_nonbenign_train = dataframes_nonbenign_train[0:l]
    
    pathtest = "/test_attack"
    test_files = gl.glob(path+pathtest+'/*.csv')
    for filename in test_files:
     
        dataframe = pd.read_csv(filename)
        dataframes_nonbenign_test = dataframes_nonbenign_test.append(dataframe)
        dataframes_nonbenign_test['class'] = 1
        
    IoT_Botnet_data_test = IoT_Botnet_benign_test.append(dataframes_nonbenign_test)
    
    IoT_Botnet_data_test = shuffle(IoT_Botnet_data_test)
    
   
    
    
    
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
    
def IoT_botnet_10_classifier():
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
    path = '/home/lehuy/DATA/shuttle/IoT'
    all_files = gl.glob(path+'/*.csv')
    for filename in all_files:
        print(filename)
        dataframe = pd.read_csv(filename)
        if 'benign' in filename:
            dataframes_benign = dataframes_benign.append(dataframe, sort= True)
            dataframes_benign['class'] = 0
        
        elif 'combo-gafgyt'in filename:
            dataframes_gafgyt_combo = dataframes_gafgyt_combo.append(dataframe, sort= True)
            dataframes_gafgyt_combo['class'] = 1
        
        elif 'junk-gafgyt' in filename:
            dataframes_gafgyt_junk = dataframes_gafgyt_junk.append(dataframe, sort= True)
            dataframes_gafgyt_junk['class'] = 2
            
        elif 'scan-gafgyt'in filename:
            dataframes_gafgyt_scan = dataframes_gafgyt_scan.append(dataframe, sort= True)
            dataframes_gafgyt_scan['class'] = 3
            
        elif 'tcp-gafgyt'in filename:
            dataframes_gafgyt_tcp = dataframes_gafgyt_tcp.append(dataframe, sort= True)
            dataframes_gafgyt_tcp['class'] = 4
            
        elif 'udp-gafgyt'in filename:
            dataframes_gafgyt_udp = dataframes_gafgyt_udp.append(dataframe, sort= True)
            dataframes_gafgyt_udp['class'] = 5
            
        elif 'ack-mirai'in filename:
            dataframes_mirai_ack = dataframes_mirai_ack.append(dataframe, sort= True)
            dataframes_mirai_ack['class'] = 6
        
        elif 'scan-mirai'in filename:
            dataframes_mirai_scan = dataframes_mirai_scan.append(dataframe, sort= True)
            dataframes_mirai_scan['class'] = 7
    
        elif 'syn-mirai'in filename:
            dataframes_mirai_syn = dataframes_mirai_syn.append(dataframe, sort= True)
            dataframes_mirai_syn['class'] = 8
        
        elif 'udp-mirai'in filename:
            dataframes_mirai_udp = dataframes_mirai_udp.append(dataframe, sort= True)
            dataframes_mirai_udp['class'] = 9
        
        elif 'udpplain-mirai' in filename:
            dataframes_mirai_udpplain = dataframes_mirai_udpplain.append(dataframe, sort= True)
            dataframes_mirai_udpplain['class'] = 10

    IoT_Botnet_data = dataframes_benign.append(dataframes_gafgyt_combo, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_junk, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_scan, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_tcp, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_gafgyt_udp, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_ack, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_scan, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_syn, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_udp, sort= True)
    IoT_Botnet_data = IoT_Botnet_data.append(dataframes_mirai_udpplain, sort= True)
    
    IoT_Botnet_data = shuffle(IoT_Botnet_data)
    IoT_Botnet_data_train = IoT_Botnet_data[153062:]
    IoT_Botnet_data_test = IoT_Botnet_data[:153062]
    
    Y_train = IoT_Botnet_data_train['class']
    Y_train = pd.get_dummies(Y_Train)
    
    Y_test = IoT_Botnet_data_test['class']
    Y_test = pd.get_dummies(Y_test)
    
    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test = IoT_Botnet_data_test.drop("class", axis=1)
    
    X_train, X_test = normalize_data(X_train, X_test, 'minmax')
    
    X_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    X_2 = X_train[IoT_Botnet_data_train["class"] == 2]
    X_3 = X_train[IoT_Botnet_data_train["class"] == 3]
    X_4 = X_train[IoT_Botnet_data_train["class"] == 4]
    X_5 = X_train[IoT_Botnet_data_train["class"] == 5]
    X_6 = X_train[IoT_Botnet_data_train["class"] == 6]
    X_7 = X_train[IoT_Botnet_data_train["class"] == 7]
    X_8 = X_train[IoT_Botnet_data_train["class"] == 8]
    X_9 = X_train[IoT_Botnet_data_train["class"] == 9]
    X_10 = X_train[IoT_Botnet_data_train["class"] == 10]
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)
    X_2 = np.array (X_2)
    X_3 = np.array (X_3)
    X_4 = np.array (X_4)
    X_5 = np.array (X_5)
    X_6 = np.array (X_6)
    X_7 = np.array (X_7)
    X_8 = np.array (X_8)
    X_9 = np.array (X_9)
    X_10 = np.array (X_10)
    return X_train, Y_train, X_test, Y_test, X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X10

#******************************************VirusShare********************************
def take_index(arr):
    array_index = []
    array_values = []
    array_X = []
    for index in range(1, len(arr), 2):
        array_index.append(arr[index])
    #print(array_index)

    for val in range(2, len(arr), 2):
        array_values.append(arr[val])
    #print(array_values)

    for i in range(0, 479):
        if str(i) in array_index:
            array_X.append(int(array_values[0]))
            del array_values[0]
        else:
            array_X.append(0)
    return array_X
def VirusShare():
    path = '/home/lehuy/DATA/dataset'
    all_files = gl.glob(path+'/*.txt')

    X = []
    Y = []

    dataframe = pd.DataFrame()
    for filename in all_files:
        data = open(filename).readlines()

        np.random.shuffle(data)
        for i in data:
            i = i.replace(':', " ").split(' ')
            Y.append(float(i[0]))
            array_X = take_index(i)
            X.append(np.array(array_X))
    Y = np.array(Y)
    Y[Y < 0.5] = 0
    Y[Y > 0.5] = 1
    dataframe = dataframe.append(pd.DataFrame(X), sort=True)
    dataframe['class'] = np.array(Y)
    print(dataframe)

    dataframe_train = dataframe[:86285]
    dataframe_test = dataframe[86285:]

    Y_train = dataframe_train['class']
    Y_train = pd.get_dummies(Y_train)

    Y_test = dataframe_test['class']
    Y_test = pd.get_dummies(Y_test)

    X_train = dataframe_train.drop('class', axis=1)
    X_test = dataframe_test.drop('class', axis=1)

    #X_train, X_test = normalize_data(X_train, X_test, 'minmax')

    X_0 = X_train[dataframe_train['class'] == 0]
    X_1 = X_train[dataframe_train['class'] == 1]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
    return X_train, Y_train, X_test, Y_test, X_0, X_1
#******************************************Antivirus********************************
def take_X(arr):
    array_X = []
    for i in range(1, 532):
        if str(i) in arr:
            array_X.append(1)
        else:
            array_X.append(0)
    return array_X
def Antivirus():
    data_train = open('/home/vuly/DATA/antivirus/dataset.train').readlines()
    data_test = open('/home/vuly/DATA/antivirus/Tst.test').readlines()
    dataframe = pd.DataFrame()
    Y = []
    X = []
    for i in data_train:
        i = i.replace(':', " ").split(' ')
        Y.append(i[0])
        array_X = take_X(i)
        X.append(np.array(array_X))

    Y = np.array(Y)
    Y[Y == '+1'] = 0
    Y[Y == '-1'] = 1
    dataframe = dataframe.append(pd.DataFrame(X))
    dataframe['class'] = np.array(Y, dtype=np.int32)
    dataframe = shuffle(dataframe)
    print(dataframe)
    dataframe_Train = dataframe[:298]
    dataframe_Test = dataframe[298:]

    X_train = dataframe_Train.drop('class', axis=1)
    Y_train = dataframe_Train['class']
    Y_train = pd.get_dummies(Y_train)

    X_test = dataframe_Test.drop('class', axis=1)
    Y_test = dataframe_Test['class']
    Y_test = pd.get_dummies(Y_test)
    X_train, X_test = normalize_data(X_train, X_test, 'minmax')

    X_0 = X_train[dataframe_Train['class'] == 0]
    X_1 = X_train[dataframe_Train['class'] == 1]
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)

    return X_train, Y_train, X_test, Y_test, X_0, X_1


    
"*****************************Load dataset*****************************"
def read_data_sets(dataset):
    path_data = "/home/cnm02/DATA/"
    NSLKDD = ["Probe", "DoS", "R2L", "U2R", "NSLKDD"]
    UNSW   = ["Fuzzers", "Analysis", "Backdoor", "DoS_UNSW", "Exploits", "Generic",\
            "Reconnaissance", "Shellcode", "Worms", "UNSW"]
    CTU13  = ["CTU13_06","CTU13_07","CTU13_08","CTU13_09","CTU13_10","CTU13_12","CTU13_13"]

    if (dataset == "nslkdd"):
        X_train, Y_train, X_test, Y_test, X0, X1 = nslkdd()
        # 13 attributes + 1 class [0 - Level0(164); level 1,2,3,4 - (139), 6 missing)
        #Some features may be CATEGORICAL, don't need to preprocessing

    elif (dataset == "unsw"):
        X_train, Y_train, X_test, Y_test, X0, X1 = unsw()

    elif (dataset == "ctu13_8"):
        X_train, Y_train, X_test, Y_test, X0, X1 = ctu13("ctu13_8")#

    elif (dataset == "ctu13_10"):
        X_train, Y_train, X_test, Y_test, X0, X1 =  ctu13("ctu13_10")
    elif (dataset == "ctu13_13"):
        X_train, Y_train, X_test, Y_test, X0, X1 =  ctu13("ctu13_13")
    elif (dataset == "VirusShare"):
        X_train, Y_train, X_test, Y_test, X0, X1 = VirusShare()
    elif (dataset == "Phishing"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Phishing()
    elif (dataset == "Antivirus"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Antivirus()
    elif (dataset == "Spam"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Spam()
    elif (dataset == "IoT"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet()
    elif (dataset == "Ads"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Internet_ads()
    else:
        print ("Incorrect data")

    
   
    class DataSets(object):
          pass
    data_sets = DataSets()
    data_sets.train = DataSet(X_train, Y_train, X0, X1)
    data_sets.test = DataSet(X_test, Y_test, X0, X1)

    return data_sets






























class DataSet(object):
  def __init__(self, features, labels,features0, features1, fake_data=False):
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
    self._features0 = features0
    self._features1 = features1
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
  @property
  def features0(self):
    return self._features0
  
  @property
  def features1(self):
    return self._features1
  
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(38)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
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

  def next_batch_100(self, batch_size):
        #major: 0
        #minor: 1
        X_ma = self.features0
        X_mi = self.features1
        
        size_class = len (X_mi)
        class_num = 2
        
        idx = np.random.randint(len(X_ma), size=size_class)
        X0 = X_ma[idx]
        Y0 = np.zeros((size_class, 1))

        idx = np.random.randint(len(X_mi), size=size_class)
        X1 = X_mi[idx]
        #eps = np.random.uniform(-0.001,0.001)
        #X1 = X1 + eps
        Y1 = np.ones((size_class, 1))

        self._features = np.concatenate((X0, X1),axis = 0)
        Y = np.concatenate((Y0, Y1),axis = 0)
        
        #convert to one-hot vector
        Y = (np.arange(class_num) ==Y[:,None]).astype(np.float32)
        Y =Y[:,0,:]
        Y = Y.astype(int)
        
        self._labels = Y

        self._num_examples = len (self._features)
        
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self._features[perm]
        self._labels = self._labels[perm]
        
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._features = self._features[perm]
          #self._features = self._features + eps
          self._labels = self._labels[perm]
          #self._cost = self._cost[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]#, self._cost[start:en
    
 
def next_batch_dss(self, batch_size, X_ma, X_mi):
        #major: 0
        #minor: 1
        size_class = len (X_mi)
        class_num = 2

        idx = np.random.randint(len(X_ma), size=size_class)
        X0 = X_ma[idx]
        Y0 = np.zeros((size_class, 1))

        idx = np.random.randint(len(X_mi), size=size_class)
        X1 = X_mi[idx]
        eps = np.random.uniform(-0.001,0.001)
        X1 = X1 + eps
        Y1 = np.ones((size_class, 1))

        self._features = np.concatenate((X0, X1),axis = 0)
        Y = np.concatenate((Y0, Y1),axis = 0)
        
        #convert to one-hot vector
        Y = (np.arange(class_num) ==Y[:,None]).astype(np.float32)
        Y =Y[:,0,:]
        Y = Y.astype(int)
        
        self._labels = Y

        self._num_examples = len (self._features)
        
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self._features[perm]
        self._labels = self._labels[perm]
        
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._features = self._features[perm]
          #self._features = self._features + eps
          self._labels = self._labels[perm]
          #self._cost = self._cost[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]#, self._cost[start:end]



