#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt 
import preprocessingdata
import operator
import os.path
import classifier
from  sklearn.metrics  import  accuracy_score , roc_curve , auc , roc_auc_score
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


datasets = np.asarray(["nslkdd", "unsw", "ctu13_8", "Ads",                       "Phishing", "Antivirus","Spam",                       "Danmini_Doorbell","Ecobee_Thermostat",                       "Ennio_Doorbell","Philips_B120N10_Baby_Monitor",                       "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera",                      "Samsung_SNH_1011_N_Webcam", "SimpleHome_XCS7_1002_WHT_Security_Camera",                       "SimpleHome_XCS7_1003_WHT_Security_Camera","nslkdd4"]) 
hidden_layers = np.asarray([[85, 30, 15], [150, 100,50],[30,20,10],[1000,500,100],
                            [50,30,15],[450,200,100],[35,20,10],[85, 30, 2],[85, 40, 15]])
batch_sizes = np.asarray([100,100,100,100,100,100,100,100,100])
data_index = 7
dataname = datasets[data_index]
hidden_layer = hidden_layers[7] #7 for IoT
batch_size = batch_sizes [7]



dt = preprocessingdata.read_data_sets(dataname)
num_sample = dt.train.num_examples
print ("size of train dataset: ",num_sample)
print ("size of test dataset: ",dt.test.num_examples)
input_dim = dt.train.features.shape[1]# mnist.train.images[0].shape[0]

balance_rate =1# len(dt.train.features1)/float(len(dt.train.features0))# mnist.train.images[0].shape[0]

label_dim = dt.train.labels.shape[1]


#nslkdd:{0: 67343, 1: 45927, 2: 52, 3: 995, 4: 11656}
print ("dimension: ",input_dim)
print  ("number of class: ",label_dim)


data_save = np.asarray([data_index, input_dim, balance_rate, label_dim])   
data_save = np.reshape(data_save, (-1,4))


if os.path.isfile("Results/datainformation.csv"): #
    auc = np.genfromtxt('Results/datainformation.csv', delimiter=',') 
    auc = np.reshape(auc,(-1,4))
    data_save = np.concatenate((auc, data_save), axis = 0)
    np.savetxt("Results/datainformation.csv", data_save,delimiter = ",",fmt = "%f")
    
else:
    np.savetxt("Results/datainformation.csv", data_save,delimiter = ",",fmt = "%f")


# In[3]:


def standalone():
    auc1, t1, TP1, FP1, TN1, FN1 = classifier.train(X_train, Y_train, X_test,Y_test, "svm")
    auc2, t2, TP2, FP2, TN2, FN2 = classifier.train(X_train, Y_train, X_test,Y_test, "pct")
    auc3, t3, TP3, FP3, TN3, FN3 = classifier.train(X_train, Y_train, X_test,Y_test, "nct")
    auc4, t4, TP4, FP4, TN4, FN4 = classifier.train(X_train, Y_train, X_test,Y_test, "lr")


    data_save = np.asarray([data_index, input_dim, balance_rate, 
                            auc1,1000 * t1,TP1, FP1, TN1, FN1,
                            auc2,1000 * t2,TP2, FP2, TN2, FN2,
                            auc3,1000 * t3, TP3, FP3, TN3, FN3 ,
                            auc4,1000 * t4, TP4, FP4, TN4, FN4])   
    data_save = np.reshape(data_save, (-1,27))


    if os.path.isfile("Results/RF_AUC_DIF/AUC_Input.csv"): #
        auc = np.genfromtxt('Results/RF_AUC_DIF/AUC_Input.csv', delimiter=',') 
        auc = np.reshape(auc,(-1,27))
        data_save = np.concatenate((auc, data_save), axis = 0)
        np.savetxt("Results/RF_AUC_DIF/AUC_Input.csv", data_save,delimiter = ",",fmt = "%f")
    
    else:
        np.savetxt("Results/RF_AUC_DIF/AUC_Input.csv", data_save,delimiter = ",",fmt = "%f")


#dt: perceptron, rf: sgd    


# In[4]:


X_train = dt.train.features
Y_train = dt.train.labels

X_test = dt.test.features
Y_test = dt.test.labels

standalone()
#auc4, t4, TP4, FP4, TN4, FN4 = classifier.train(X_train, Y_train, X_test,Y_test, "rf12")


# In[5]:


#AutoEncoder code
class AE(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, hidden_layers = [85, 30,12]):
       
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.build()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])
        
        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, self.hidden_layers[0], scope='ae_enc_fc1', activation_fn=tf.nn.relu)
        #f2 = fc(f1, 60, scope='enc_fc2', activation_fn=tf.nn.tanh)
        f3 = fc(f1, self.hidden_layers[1], scope='ae_enc_fc3', activation_fn=tf.nn.relu)
        #f4 = fc(f3, 20, scope='enc_fc4', activation_fn=tf.nn.relu)
        
        
        self.z = fc(f3, self.hidden_layers[2], scope='ae_enc_fc5_mu', activation_fn=None)
       
        # Decode
        # z,y -> x_hat
        # g1 = fc(self.Z, 20, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(self.z,self.hidden_layers[1], scope='ae_dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, self.hidden_layers[0], scope='ae_dec_fc3', activation_fn=tf.nn.relu)
        #g4 = fc(g3, 85, scope='dec_fc4', activation_fn=tf.nn.tanh)
       
        self.x_hat = fc(g3, input_dim, scope='ae_dec_fc5', activation_fn=tf.sigmoid)
        #self.x_res = self.x_hat[:,0:input_dim]

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        recon_loss = tf.reduce_mean(tf.square(self.x - self.x_hat),1) #(((self.x - y)**2).mean(1)).mean()
        self.recon_loss = tf.reduce_mean(recon_loss)

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.recon_loss)
        
        
       
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
       
        _, recon_loss = self.sess.run(
            [self.train_op,  self.recon_loss],
            feed_dict={self.x: x}
           
        )
      
        return recon_loss
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


# In[8]:


#VAE code
class VAE(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, hidden_layers = [85, 30,12]):
       
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.build()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])
        
        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, self.hidden_layers[0], scope='vae_enc_fc1', activation_fn=tf.nn.relu)
        #f2 = fc(f1, 60, scope='enc_fc2', activation_fn=tf.nn.tanh)
        f3 = fc(f1, self.hidden_layers[1], scope='vae_enc_fc3', activation_fn=tf.nn.relu)
        #f4 = fc(f3, 20, scope='enc_fc4', activation_fn=tf.nn.relu)
        
        
        self.z_mu = fc(f3, self.hidden_layers[2], scope='vae_enc_fc5_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.hidden_layers[2], scope='vae_enc_fc5_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z,y -> x_hat
        # g1 = fc(self.Z, 20, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(self.z,self.hidden_layers[1], scope='vae_dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, self.hidden_layers[0], scope='vae_dec_fc3', activation_fn=tf.nn.relu)
        #g4 = fc(g3, 85, scope='dec_fc4', activation_fn=tf.nn.tanh)
       
        self.x_hat = fc(g3, input_dim, scope='vae_dec_fc5', activation_fn=tf.sigmoid)
        #self.x_res = self.x_hat[:,0:input_dim]

        # Loss
        # Reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(self.x - self.x_hat),1) #(((self.x - y)**2).mean(1)).mean()

        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        
        #original
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq) , axis=1)
        
        
        self.latent_loss =  tf.reduce_mean(latent_loss)
        self.total_loss = tf.reduce_mean(recon_loss +latent_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        
        
       
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
       
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
           
        ) 
        return loss, recon_loss, latent_loss
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


# In[ ]:


#MAE code:
#We are organizing this code


# In[ ]:


#MDAE code:
#We are organizing this code


# In[ ]:


#MVAE code:
#We are organizing this code


# In[15]:



MVAE_loss_ = []
MVAE_recon_loss_=[]
MVAE_latent_loss_ = []
MVAE_auc_svm_ = []
MVAE_auc_dt_ = []
MVAE_auc_rf_ = []
MVAE_auc_lr_ = []

MVAE_t1_ = []
MVAE_t2_ = []
MVAE_t3_ = []
MVAE_t4_ = []

MVAE_TP_1 = []
MVAE_FP_1 = []
MVAE_TN_1 = []
MVAE_FN_1 = []

MVAE_TP_2 = []
MVAE_FP_2 = []
MVAE_TN_2 = []
MVAE_FN_2 = []

MVAE_TP_3 = []
MVAE_FP_3 = []
MVAE_TN_3 = []
MVAE_FN_3 = []

MVAE_TP_4 = []
MVAE_FP_4 = []
MVAE_TN_4 = []
MVAE_FN_4 = []
          
def MVAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [80, 30, 15]):
    path = "Results/hidden/MVAE/"
    model = MVAE(learning_rate=learning_rate,
                                    batch_size=batch_size, hidden_layers=hidden_layers) 
    X_train = dt.train.features
    Y_train = dt.train.labels

    X_test = dt.test.features
    Y_test = dt.test.labels
    for epoch in range(num_epoch):
       
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, y_mb = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            
            loss, recon_loss, latent_loss= model.run_single_step(X_mb, y_mb)
         
        if epoch % step == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model.transformer(X_train)
            print (np.isnan(z_train))
            s = time.time()
            z_test = model.transformer(X_test)
            print (np.isnan(z_test))
            e = time.time()
            t_tr = (e - s)/ float(len(X_test))
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            

            auc_svm, t1, TP1, FP1, TN1, FN1 = classifier.train(z_train, Y_train, z_test,Y_test, "svm")
            auc_dt, t2, TP2, FP2, TN2, FN2 = classifier.train(z_train, Y_train, z_test,Y_test, "dt")
            auc_rf, t3, TP3, FP3, TN3, FN3 = classifier.train(z_train, Y_train, z_test,Y_test, "rf")
            auc_lr, t4, TP4, FP4, TN4, FN4 = classifier.train(z_train, Y_train, z_test,Y_test, "lr")
           
           
            MVAE_loss_.append(loss)
            MVAE_recon_loss_.append(recon_loss)
            MVAE_latent_loss_.append(latent_loss) 
            MVAE_auc_svm_.append(auc_svm)
            MVAE_auc_dt_.append(auc_dt)
            MVAE_auc_rf_.append(auc_rf)
            MVAE_auc_lr_.append(auc_lr)
            MVAE_t1_.append(t1 + t_tr)
            MVAE_t2_.append(t2 + t_tr)
            MVAE_t3_.append(t3 + t_tr)
            MVAE_t4_.append(t4 + t_tr)
            
            MVAE_TP_1.append(TP1)
            MVAE_FP_1.append(FP1)
            MVAE_TN_1.append(TN1)
            MVAE_FN_1.append(FN1)
            
            MVAE_TP_2.append(TP2)
            MVAE_FP_2.append(FP2)
            MVAE_TN_2.append(TN2)
            MVAE_FN_2.append(FN2)
     
    
            MVAE_TP_3.append(TP3)
            MVAE_FP_3.append(FP3)
            MVAE_TN_3.append(TN3)
            MVAE_FN_3.append(FN3)
     
    
            MVAE_TP_4.append(TP4)
            MVAE_FP_4.append(FP4)
            MVAE_TN_4.append(TN4)
            MVAE_FN_4.append(FN4)
     
     
    print('Done MVAE!')
    return model


# In[16]:


def MVAE_trainer_forvisulizing(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [80, 30, 15]):
    path = "Results/hidden/MVAE_v/"
    model = MVAE(learning_rate=learning_rate,
                                    batch_size=batch_size, hidden_layers=hidden_layers) 
    #X_train = dt.train.features
    #Y_train = dt.train.labels

    #X_test = dt.test.features
    #Y_test = dt.test.labels
    num_sample = len (X_train)
    for epoch in range(num_epoch):
       
        
        for iter in range(num_sample // batch_size):
           
            X_mb, y_mb = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            
            loss, recon_loss, latent_loss= model.run_single_step(X_mb, y_mb)
         
        if epoch % step == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model.transformer(X_train)
          
            z_test = model.transformer(X_test)
           
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_test, delimiter=",", fmt='%f' )
            

     
     
    print('Done MVAE!')
    return model


# In[17]:


AE_recon_loss_=[]
AE_auc_svm_ = []
AE_auc_dt_ = []
AE_auc_rf_ = []
AE_auc_lr_ = []

AE_t1_ = []
AE_t2_ = []
AE_t3_ = []
AE_t4_ = []

AE_TP_1 = []
AE_FP_1 = []
AE_TN_1 = []
AE_FN_1 = []

AE_TP_2 = []
AE_FP_2 = []
AE_TN_2 = []
AE_FN_2 = []

AE_TP_3 = []
AE_FP_3 = []
AE_TN_3 = []
AE_FN_3 = []

AE_TP_4 = []
AE_FP_4 = []
AE_TN_4 = []
AE_FN_4 = []

def AE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2]):
    path = "Results/hidden/AE/"
    model1 = AE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers)
    for epoch in range(num_epoch):
        
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss= model1.run_single_step(X_mb)
        
        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}'.format(
                epoch, recon_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model1.transformer(X_train)
            s = time.time()
            z_test = model1.transformer(X_test)
            e = time.time()
            t_tr = (e-s)/float(len(X_test))
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_test, delimiter=",", fmt='%f' )
            auc_svm, t1, TP1, FP1, TN1, FN1 = classifier.train(z_train, Y_train, z_test,Y_test, "svm")
            auc_dt, t2, TP2, FP2, TN2, FN2 = classifier.train(z_train, Y_train, z_test,Y_test, "dt")
            auc_rf, t3, TP3, FP3, TN3, FN3 = classifier.train(z_train, Y_train, z_test,Y_test, "rf")
            auc_lr, t4, TP4, FP4, TN4, FN4 = classifier.train(z_train, Y_train, z_test,Y_test, "lr")
           
           
            AE_recon_loss_.append(recon_loss)
          
            AE_auc_svm_.append(auc_svm)
            AE_auc_dt_.append(auc_dt)
            AE_auc_rf_.append(auc_rf)
            AE_auc_lr_.append(auc_lr)
            AE_t1_.append(t1 + t_tr)
            AE_t2_.append(t2 + t_tr)
            AE_t3_.append(t3 + t_tr)
            AE_t4_.append(t4 + t_tr)
            
            AE_TP_1.append(TP1)
            AE_FP_1.append(FP1)
            AE_TN_1.append(TN1)
            AE_FN_1.append(FN1)
            
            AE_TP_2.append(TP2)
            AE_FP_2.append(FP2)
            AE_TN_2.append(TN2)
            AE_FN_2.append(FN2)
            
            AE_TP_3.append(TP3)
            AE_FP_3.append(FP3)
            AE_TN_3.append(TN3)
            AE_FN_3.append(FN3)
            
            AE_TP_4.append(TP4)
            AE_FP_4.append(FP4)
            AE_TN_4.append(TN4)
            AE_FN_4.append(FN4)
       
    print('Done AE!')
    
   
    return model1


# In[18]:


def AE_trainer_forvisualizing(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2]):
    path = "Results/hidden/AE/"
    model1 = AE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers)
    for epoch in range(num_epoch):
        
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss= model1.run_single_step(X_mb)
        
        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}'.format(
                epoch, recon_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model1.transformer(X_train)
            
            z_test = model1.transformer(X_test)
           
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_test, delimiter=",", fmt='%f' )
    print('Done AE!') 
    return model1


# In[19]:


MAE_recon_loss_= []
MAE_multi_loss_ = []
MAE_auc_svm_ = []
MAE_auc_dt_ = []
MAE_auc_rf_ = []
MAE_auc_lr_ = []

MAE_t1_ = []
MAE_t2_ = []
MAE_t3_ = []
MAE_t4_ = []

MAE_TP_1 = []
MAE_FP_1 = []
MAE_TN_1 = []
MAE_FN_1 = []

MAE_TP_2 = []
MAE_FP_2 = []
MAE_TN_2 = []
MAE_FN_2 = []

MAE_TP_3 = []
MAE_FP_3 = []
MAE_TN_3 = []
MAE_FN_3 = []

MAE_TP_4 = []
MAE_FP_4 = []
MAE_TN_4 = []
MAE_FN_4 = []

def MAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2]):
    path = "Results/hidden/MAE/"
    model1 = MAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers)
    for epoch in range(num_epoch):
        
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, y_mb = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss, multi_loss= model1.run_single_step(X_mb, y_mb)
        
        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}  Multi loss: {}'.format(
                epoch, recon_loss, multi_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model1.transformer(X_train)
            s = time.time()
            z_test = model1.transformer(X_test)
            e = time.time()
            t_tr = (e-s)/float(len(X_test))
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            auc_svm, t1, TP1, FP1, TN1, FN1 = classifier.train(z_train, Y_train, z_test,Y_test, "svm")
            auc_dt, t2, TP2, FP2, TN2, FN2 = classifier.train(z_train, Y_train, z_test,Y_test, "dt")
            auc_rf, t3, TP3, FP3, TN3, FN3 = classifier.train(z_train, Y_train, z_test,Y_test, "rf")
            auc_lr, t4, TP4, FP4, TN4, FN4 = classifier.train(z_train, Y_train, z_test,Y_test, "lr")
                  
            MAE_recon_loss_.append(recon_loss)
            MAE_multi_loss_.append(multi_loss)
          
            MAE_auc_svm_.append(auc_svm)
            MAE_auc_dt_.append(auc_dt)
            MAE_auc_rf_.append(auc_rf)
            MAE_auc_lr_.append(auc_lr)
            MAE_t1_.append(t1 + t_tr)
            MAE_t2_.append(t2 + t_tr)
            MAE_t3_.append(t3 + t_tr)
            MAE_t4_.append(t4 + t_tr)
            
            MAE_TP_1.append(TP1)
            MAE_FP_1.append(FP1)
            MAE_TN_1.append(TN1)
            MAE_FN_1.append(FN1)
            
            MAE_TP_2.append(TP2)
            MAE_FP_2.append(FP2)
            MAE_TN_2.append(TN2)
            MAE_FN_2.append(FN2)
            
            MAE_TP_3.append(TP3)
            MAE_FP_3.append(FP3)
            MAE_TN_3.append(TN3)
            MAE_FN_3.append(FN3)
            
            MAE_TP_4.append(TP4)
            MAE_FP_4.append(FP4)
            MAE_TN_4.append(TN4)
            MAE_FN_4.append(FN4)
       
    print('Done MAE!')
    
   
    return model1


# In[20]:


def MAE_trainer_forvisualizing(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2]):
    path = "Results/hidden/MAE_v/"
    model1 = MAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers)
    for epoch in range(num_epoch):
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, y_mb = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss, multi_loss= model1.run_single_step(X_mb, y_mb)
        
        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}  Multi loss: {}'.format(
                epoch, recon_loss, multi_loss))
           
               
            z_train = model1.transformer(X_train)
           
            z_test = model1.transformer(X_test)
          
           
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_test, delimiter=",", fmt='%f' )
           
       
    print('Done MAE to visualize!')
    
   
    return model1


# In[21]:


DMAE_recon_loss_= []
DMAE_multi_loss_ = []
DMAE_auc_svm_ = []
DMAE_auc_dt_ = []
DMAE_auc_rf_ = []
DMAE_auc_lr_ = []

DMAE_t1_ = []
DMAE_t2_ = []
DMAE_t3_ = []
DMAE_t4_ = []

DMAE_TP_1 = []
DMAE_FP_1 = []
DMAE_TN_1 = []
DMAE_FN_1 = []

DMAE_TP_2 = []
DMAE_FP_2 = []
DMAE_TN_2 = []
DMAE_FN_2 = []

DMAE_TP_3 = []
DMAE_FP_3 = []
DMAE_TN_3 = []
DMAE_FN_3 = []

DMAE_TP_4 = []
DMAE_FP_4 = []
DMAE_TN_4 = []
DMAE_FN_4 = []

def DMAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2], noise_factor = 0.0025):
    path = "Results/hidden/DMAE/"
    model1 = DMAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers, noise_factor = noise_factor)
    for epoch in range(num_epoch):
        
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, y_mb = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss, multi_loss= model1.run_single_step(X_mb, y_mb)
        
        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}  Multi loss: {}'.format(
                epoch, recon_loss, multi_loss))
            #model.writer.add_summary(summary, epoch )
               
            z_train = model1.transformer(X_train)
            s = time.time()
            z_test = model1.transformer(X_test)
            e = time.time()
            t_tr = (e-s)/float(len(X_test))
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            auc_svm, t1, TP1, FP1, TN1, FN1 = classifier.train(z_train, Y_train, z_test,Y_test, "svm")
            auc_dt, t2, TP2, FP2, TN2, FN2 = classifier.train(z_train, Y_train, z_test,Y_test, "dt")
            auc_rf, t3, TP3, FP3, TN3, FN3 = classifier.train(z_train, Y_train, z_test,Y_test, "rf")
            auc_lr, t4, TP4, FP4, TN4, FN4 = classifier.train(z_train, Y_train, z_test,Y_test, "lr")
             
           
           
            DMAE_recon_loss_.append(recon_loss)
            DMAE_multi_loss_.append(multi_loss)
          
            DMAE_auc_svm_.append(auc_svm)
            DMAE_auc_dt_.append(auc_dt)
            DMAE_auc_rf_.append(auc_rf)
            DMAE_auc_lr_.append(auc_lr)
            DMAE_t1_.append(t1 + t_tr)
            DMAE_t2_.append(t2 + t_tr)
            DMAE_t3_.append(t3 + t_tr)
            DMAE_t4_.append(t4 + t_tr)
            
            DMAE_TP_1.append(TP1)
            DMAE_FP_1.append(FP1)
            DMAE_TN_1.append(TN1)
            DMAE_FN_1.append(FN1)
            
            DMAE_TP_2.append(TP2)
            DMAE_FP_2.append(FP2)
            DMAE_TN_2.append(TN2)
            DMAE_FN_2.append(FN2)
            
            DMAE_TP_3.append(TP3)
            DMAE_FP_3.append(FP3)
            DMAE_TN_3.append(TN3)
            DMAE_FN_3.append(FN3)
            
            DMAE_TP_4.append(TP4)
            DMAE_FP_4.append(FP4)
            DMAE_TN_4.append(TN4)
            DMAE_FN_4.append(FN4)
       
    print('Done DMAE!')
    
   
    return model1


# In[22]:


VAE_loss_=[]
VAE_recon_loss_=[]
VAE_latent_loss_=[]
VAE_auc_svm_ = []
VAE_auc_dt_ = []
VAE_auc_rf_ = []
VAE_auc_lr_ = []
VAE_t1_ = []
VAE_t2_ = []
VAE_t3_ = []
VAE_t4_ = []

VAE_TP_1 = []
VAE_FP_1 = []
VAE_TN_1 = []
VAE_FN_1 = []

VAE_TP_2 = []
VAE_FP_2 = []
VAE_TN_2 = []
VAE_FN_2 = []

VAE_TP_3 = []
VAE_FP_3 = []
VAE_TN_3 = []
VAE_FN_3 = []

VAE_TP_4 = []
VAE_FP_4 = []
VAE_TN_4 = []
VAE_FN_4 = []

def VAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers = [7,4,2]):
    path = "Results/hidden/VAE/"
    
    model = VAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers)
    X_train = dt.train.features
    Y_train = dt.train.labels

    X_test = dt.test.features
    Y_test = dt.test.labels
    for epoch in range(num_epoch):
        
        num_sample = len (X_train)
        for iter in range(num_sample // batch_size):
           
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            loss, recon_loss, latent_loss= model.run_single_step(X_mb)
         
        if epoch % step == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))
               
            z_train = model.transformer(X_train)
            s = time.time()
            z_test = model.transformer(X_test)
            e = time.time()
            t_tr = (e-s)/len(X_test)
            np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            auc_svm, t1, TP1, FP1, TN1, FN1 = classifier.train(z_train, Y_train, z_test,Y_test, "svm")
            auc_dt, t2, TP2, FP2, TN2, FN2 = classifier.train(z_train, Y_train, z_test,Y_test, "dt")
            auc_rf, t3, TP3, FP3, TN3, FN3 = classifier.train(z_train, Y_train, z_test,Y_test, "rf")
            auc_lr, t4, TP4, FP4, TN4, FN4 = classifier.train(z_train, Y_train, z_test,Y_test, "lr")
           
            VAE_loss_.append(loss)
            VAE_recon_loss_.append(recon_loss)
            VAE_latent_loss_.append(latent_loss)
          
            VAE_auc_svm_.append(auc_svm)
            VAE_auc_dt_.append(auc_dt)
            VAE_auc_rf_.append(auc_rf)
            VAE_auc_lr_.append(auc_lr)
            VAE_t1_.append(t1 + t_tr)
            VAE_t2_.append(t2 + t_tr)
            VAE_t3_.append(t3 + t_tr)
            VAE_t4_.append(t4 + t_tr)
            
            VAE_TP_1.append(TP1)
            VAE_FP_1.append(FP1)
            VAE_TN_1.append(TN1)
            VAE_FN_1.append(FN1)
            
            VAE_TP_2.append(TP2)
            VAE_FP_2.append(FP2)
            VAE_TN_2.append(TN2)
            VAE_FN_2.append(FN2)
            
            VAE_TP_3.append(TP3)
            VAE_FP_3.append(FP3)
            VAE_TN_3.append(TN3)
            VAE_FN_3.append(FN3)
            
            VAE_TP_4.append(TP4)
            VAE_FP_4.append(FP4)
            VAE_TN_4.append(TN4)
            VAE_FN_4.append(FN4)
       
    print('Done VAE!')
    
    return model


# In[23]:


# Compute accuracy for a given set of predictions and labels
def compute_auc(labels,preds):
   return roc_auc_score(labels, preds)


# In[ ]:





# In[25]:


def save_auc_tofile(method, svms,t1s,svm_tp, svm_fp, svm_tn, svm_fn,
                    dts,t2s,dt_tp, dt_fp, dt_tn, dt_fn,
                    rfs,t3s,rf_tp, rf_fp, rf_tn, rf_fn,
                    lrs,t4s,lr_tp, lr_fp, lr_tn, lr_fn):
    ep = 500
    step = 20
    ep = ep / step
    idx,auc1 = max(enumerate(svms), key=operator.itemgetter(1))
    TP1 = svm_tp[idx]
    FP1 = svm_fp[idx]
    TN1 = svm_tn[idx]
    FN1 = svm_fn[idx]
    auc2 = dts[idx]
    TP2 = dt_tp[idx]
    FP2 = dt_fp[idx]
    TN2 = dt_tn[idx]
    FN2 = dt_fn[idx]
    auc3 = rfs[idx]
    TP3 = rf_tp[idx]
    FP3 = rf_fp[idx]
    TN3 = rf_tn[idx]
    FN3 = rf_fn[idx]
    auc4 = lrs [idx]
    TP4 = lr_tp[idx]
    FP4 = lr_fp[idx]
    TN4 = lr_tn[idx]
    FN4 = lr_fn[idx]
    
    t1 =1000 * sum(t1s)/len(t1s)
    t2 =1000 *  sum(t2s)/len(t2s)
    t3 =1000 *  sum(t3s)/len(t3s)
    t4 =1000 *  sum(t4s)/len(t4s)
    #for unknown
    data_index = 100
    data_save = np.asarray([data_index, input_dim, balance_rate, 
                        auc1,1000 * t1,TP1, FP1, TN1, FN1,
                        auc2,1000 * t2,TP2, FP2, TN2, FN2,
                        auc3,1000 * t3, TP3, FP3, TN3, FN3 ,
                        auc4,1000 * t4, TP4, FP4, TN4, FN4])   
    data_save = np.reshape(data_save, (-1,27))


    if os.path.isfile("Results/RF_AUC_DIF/AUC_Hidden_"+method+".csv"): #
        auc = np.genfromtxt("Results/RF_AUC_DIF/AUC_Hidden_"+method+".csv", delimiter=',') 
        auc = np.reshape(auc,(-1,27))
        data_save = np.concatenate((auc, data_save), axis = 0)
        np.savetxt("Results/RF_AUC_DIF/AUC_Hidden_"+method+".csv", data_save,delimiter = ",",fmt = "%f")
    
    else:
        np.savetxt("Results/RF_AUC_DIF/AUC_Hidden_"+method+".csv", data_save,delimiter = ",",fmt = "%f")
    return 1


# In[26]:


# Train the model
num_epoch = 101
step = 20
#hidden_layers = np.asarray([[85, 30, 15], [80, 40, 20],[29,18,7],[29,18,7],[29,18,7],[0,0,0],
#                         [20,12,8],[450,200,100],[35,20,10],[0,0,0],[1000,500,100]])
filter_sizes = np.asarray([[1,2,3],[3,5,7],[1,2,3],[7,11,15],[1,2,3],[3,5,7],[1,2,3],[1,2,3]])
data_shapes = np.asarray ([[12,12],[14,14],[8,8],[40,40],[9,9],[25,25],[8,8],[11,11]])
#hidden_layers = [29,18,7] #ctu13 co 40 attributes
#hidden_layers = [45,28,16]#spam
#hidden_layers = [85,55,20]#
hidden_layer = hidden_layers[7]
block_size = batch_sizes[7]
lr = 1e-4
noise_factor = 0.001# 0.0025   #0, 0.0001, 0,001, 0.01, 0.1, 1.0
filter_size = filter_sizes[0]#[1,2,3]
data_shape = data_shapes[0] #[12,12]
#conf = str(num_epoch)+"_"+str(block_size)+"_"+str (lr)+"_"+str (hidden_layer[0])+"_"+str (hidden_layer[1])+"_"+str (hidden_layer[2])+"_noise: "+str(noise_factor)

model = AE_trainer_v(learning_rate=lr, batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)

#model = AE_trainer(learning_rate=lr, batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)
#save auc to file:

#save_auc_tofile("AE", AE_auc_svm_, AE_t1_,AE_TP_1,AE_FP_1,AE_TN_1,AE_FN_1,
#                AE_auc_dt_, AE_t2_,AE_TP_2,AE_FP_2,AE_TN_2,AE_FN_2,
#                AE_auc_rf_, AE_t3_,AE_TP_3,AE_FP_3,AE_TN_3,AE_FN_3,
#               AE_auc_lr_, AE_t4_,AE_TP_4,AE_FP_4,AE_TN_4,AE_FN_4)





#model = MAE_trainer_v(learning_rate=lr,  batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)
#model = MAE_trainer(learning_rate=lr,  batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)

#save auc to file:

#save_auc_tofile("MAE", MAE_auc_svm_, MAE_t1_,MAE_TP_1,MAE_FP_1,MAE_TN_1,MAE_FN_1,
#                MAE_auc_dt_, MAE_t2_,MAE_TP_2,MAE_FP_2,MAE_TN_2,MAE_FN_2,
#                MAE_auc_rf_, MAE_t3_,MAE_TP_3,MAE_FP_3,MAE_TN_3,MAE_FN_3,
#                MAE_auc_lr_, MAE_t4_,MAE_TP_4,MAE_FP_4,MAE_TN_4,MAE_FN_4)



#model = DMAE_trainer(learning_rate=lr,  batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer, noise_factor = noise_factor)

#save auc to file:

#save_auc_tofile("DMAE", DMAE_auc_svm_, DMAE_t1_,DMAE_TP_1,DMAE_FP_1,DMAE_TN_1,DMAE_FN_1,
#                DMAE_auc_dt_, DMAE_t2_,DMAE_TP_2,DMAE_FP_2,DMAE_TN_2,MAE_FN_2,
#                DMAE_auc_rf_, DMAE_t3_,DMAE_TP_3,DMAE_FP_3,DMAE_TN_3,DMAE_FN_3,
#                DMAE_auc_lr_, DMAE_t4_,DMAE_TP_4,DMAE_FP_4,DMAE_TN_4,DMAE_FN_4)


#model = MVAE_trainer(learning_rate=lr,  batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)

#save auc to file:
#save_auc_tofile("MVAE", MVAE_auc_svm_, MVAE_t1_,MVAE_TP_1, MVAE_FP_1, MVAE_TN_1, MVAE_FN_1,
#                MVAE_auc_dt_, MVAE_t2_,MVAE_TP_2,MVAE_FP_2,MVAE_TN_2,MVAE_FN_2,
#                MVAE_auc_rf_, MVAE_t3_,MVAE_TP_3,MVAE_FP_3,MVAE_TN_3,MVAE_FN_3,
#                MVAE_auc_lr_, MVAE_t4_,MVAE_TP_4,MVAE_FP_4,MVAE_TN_4,MVAE_FN_4)


#model = MVAE_trainer_v(learning_rate=lr,  batch_size=block_size, num_epoch=num_epoch, hidden_layers = hidden_layer)



#Save Y for visualizing
Y_train = np.argmax (Y_train, 1)
Y_test = np.argmax (Y_test, 1)

np.savetxt('Results/hidden/AE/Y_train.txt', Y_train, delimiter=",")
np.savetxt('Results/hidden/AE/Y_test.txt', Y_test, delimiter=",")


# In[ ]:


header = "epoch,DMVAE_loss_,DMVAE_recon_loss_, DMVAE_latent_loss_,DMVAE_auc_svm_,DMVAE_auc_dt_,DMVAE_auc_rf_,
D_dss_loss, D_dss_RE, D_dss_KL, D_dss_svm, D_dss_dt, D_dss_rf, 
mvae_loss, mvae_RE, mvae_KL, mvae_svm, mvae_dt, mvae_rf, 
vae_loss, vae_RE, vae_KL, vae_svm, vae_dt, vae_rf,
ae_loss, ae_svm, ae_dt, ae_rf,
CNN_auc_svm_, CNN_auc_dt_, CNN_auc_rf_,
MAE_recon_loss_,MAE_multi_loss_,MAE_auc_svm_,MAE_auc_pen_,MAE_auc_cen_,
DMAE_recon_loss_,DMAE_multi_loss_,DMAE_auc_svm_,DMAE_auc_pen_,DMAE_auc_cen_,DMAE_auc_lr_"
t = np.arange(0, num_epoch, step)  
print (len(t))
print (len(DMVAE_loss_))
DSS_loss_ = [0.5] * len(AE_recon_loss_)
DSS_recon_loss_ = [0.5] * len(AE_recon_loss_)
DSS_latent_loss_ = [0.5] * len(AE_recon_loss_)

DSS_auc_svm_ = [0.5] * len(AE_recon_loss_)
DSS_auc_dt_ = [0.5] * len(AE_recon_loss_)
DSS_auc_rf_ = [0.5] * len(AE_recon_loss_)

np.savetxt(dataname+"_"+conf+ ".csv", np.column_stack((t,DMVAE_loss_,DMVAE_recon_loss_, DMVAE_latent_loss_,DMVAE_auc_svm_,DMVAE_auc_dt_,DMVAE_auc_rf_,
                                                       DSS_loss_,DSS_recon_loss_,DSS_latent_loss_,DSS_auc_svm_,DSS_auc_dt_,DSS_auc_rf_, 
                                                       MVAE_loss_,MVAE_recon_loss_, MVAE_latent_loss_,MVAE_auc_svm_,MVAE_auc_dt_,MVAE_auc_rf_,
                                                       VAE_loss_,VAE_recon_loss_, VAE_latent_loss_,VAE_auc_svm_,VAE_auc_dt_,VAE_auc_rf_,
                                                       AE_recon_loss_,AE_auc_svm_,AE_auc_dt_,AE_auc_rf_,
                                                       CNN_auc_svm_, CNN_auc_dt_, CNN_auc_rf_,
                                                       MAE_recon_loss_,MAE_multi_loss_,MAE_auc_svm_,MAE_auc_dt_,MAE_auc_rf_)),
           delimiter=",", fmt='%s', header=header)

