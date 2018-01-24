#
#import necessary libraries
#
import glob #file path names
import os #os dependent functions
import librosa #library for sound analysis
from pyAudioAnalysis import audioBasicIO #another lib for sound analysis
from pyAudioAnalysis import audioFeatureExtraction #works only under python2
import numpy as np #math functions
import matplotlib.pyplot as plt #plotting
import tensorflow as tf #google "neural network" (NN), can also use sklearn, keras or other
#if tensorflow gives an error about python 3.5-3.6, use <pip install tf-nightly>

#cd to target folder N.B. don't use this if you run this as a stand-alone program
os.chdir("/path_to_my_folder")

#conventional telephone line covers 300Hz-3400Hz and uses 8000Hz sampling frequency
#VoIP uses WidebandAudio that covers 50Hz-7000Hz (or higher) and uses 16000Hz sampling frequency
#Skype and other popular platforms are actually using anything between 8kHz and 24kHz depending on connection quality 

#except 44.1kHz, all files are band-pass filtered and downsampled in praat using
#frequency-domain filtering (i.e. fft) and Hann window (see band_pass_filter_resample.praat)
#python has scipy.signal.butter which does Butterworth filter, but no straightforward Hann window function

#
#define functions for feature extraction
#

#function to extract acoustic features
#we will use this to get features that nn can work with
#these are based on various acoustic measures
def extract_acoustic_features(file_name):
    X, sample_rate = librosa.load(file_name)
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #mfcc: Mel-frequency cepstral coefficients
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #melspectrogram: Compute a Mel-scaled power spectrogram, mel = 2595 * log10 (1 + hertz / 700)
    stft = np.abs(librosa.stft(X)) #short-time Fourier transform for features below
    #the following makes more sense for music and harmonic sounds (i.e. vowels)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) #chorma-stft: Compute a chromagram from a waveform or power spectrogram
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) #spectral_contrast: Compute spectral contrast
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0) #tonnetz: Computes the tonal centroid
    return mfcc,chroma,mel,contrast,tonnetz

#function to process audio files
#open files in each folder
#extract features and get category labels from file names
def process_files(recordings_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,261)), np.empty(0) #193 is the number of features we get from extract_acoustic_features + 68 from pyAudioAnalysis
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(recordings_dir, sub_dir, file_ext)):
            #get features using pyAudioAnalysis
            [Fs, x] = audioBasicIO.readAudioFile(fn) #open file for pyAudioAnalysis
            #Mid-term feature extraction - M, SD of short-term feature sequence (essentially stft, but we define the window)
            #mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep), also extracts short-term features, but we are not using them
            [mtF, stF] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 1*Fs, 1*Fs, 0.050*Fs, 0.025*Fs) #50ms frame size with 25ms frame step (50% overlap) for stft
            mtF = np.reshape(mtF,-1) #get rid of one empty dimension, so we can np.hstack below
            #get features using librosa via extract_acoustic_features
            mfcc, chroma, mel, contrast,tonnetz = extract_acoustic_features(fn) #extract features here
            get_features = np.hstack([mfcc,chroma,mel,contrast,tonnetz, mtF]) #we need all numbers in one vector so np.hstack
#            print(mfcc, chroma, mel, contrast,tonnetz, mtF) #look at each acoustic measure
            features = np.vstack([features,get_features]) #each file is one row, so np.vstack
            labels = np.append(labels, fn.split('/')[2].split('_')[2]) #class labels come from file names 006_food_1_.wav gives 1
            print(fn) #print file names as they get processed        #split '/', then by '_', take the third element (after the seond '/' and '_')
    return np.array(features), np.array(labels, dtype = np.int)

#function to dummy code categories a.k.a one_hot_encode
#N.B. since label processing and dummy coding is done using numpy, the categories need to be numeric
def one_hot_encode(labels):
    n_labels = len(labels) #overall n
    n_unique_labels = len(np.unique(labels)) #n of unique  categories
    one_hot_encode = np.zeros((n_labels,n_unique_labels)) #populate with zeros
    one_hot_encode[np.arange(n_labels), labels-1] = 1 #or labels] #each column will have only zeros and ones
    return one_hot_encode

#
#extract acoustic features
#

#define folders
recordings_dir = 'recordings'
train_dir = ['train_tokens_16kHz']
test_dir = ['test_tokens_16kHz']

#extract features
train_features, train_labels = process_files(recordings_dir,train_dir)
test_features, test_labels = process_files(recordings_dir,test_dir)

#dummy code categories a.k.a one_hot_encode
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

#save extracted features (to be used in orange or with other algorithms)
#np.savetxt("train_features.csv", np.column_stack([train_labels,train_features]), delimiter=",")
#np.savetxt("test_features.csv", np.column_stack((test_labels,test_features)), delimiter=",")

#load saved csv
#train = np.genfromtxt('train_features.csv',delimiter=",") #load csv
#train_labels = train[:,0:3] #select first 3 columns containing our labels
#train_features = train[:,3:] #select the rest for features
#test = np.genfromtxt('test_features.csv',delimiter=",")
#test_labels = test[:,0:3]
#test_features = test[:,3:]

#
#setup simple nn for classification/categorization task
#essentially a multilayer perceptron (MLP)
#

#define some nn parameters
n_runs = 100 #n of training epochs or steps
n_dim = train_features.shape[1] #data set size
n_classes = 3 #n of classes to be categorized
n_hidden_1 = 100 #n of units in the first hidden layer
n_hidden_2 = 100 #n of units in the second hidden layer
sd = 1 / np.sqrt(n_dim) #can do n-1
learning_rate = 0.01 #no decay applied here

#define graph input (N.B. tf is using graph based computations)
train_inputs = tf.placeholder(tf.float32,[None,n_dim]) #placeholders for features 
train_outputs = tf.placeholder(tf.float32,[None,n_classes]) #placeholders for class labels 

#define layer weights, biases, and activation function
#for output layer use sigmoid or tanh activation function for two categories
#use softmax when there are more than two categories (we have three talkers)
#for in-between layer nonlinearity use relu (rectified linear unit function, a newer algorithm) or sigmoid (more traditional)
#
#to get more control (especially over activation functions), we define each layer's parameters separately

#first hidden layer
w_h1 = tf.Variable(tf.random_normal([n_dim,n_hidden_1], mean = 0, stddev=sd)) #weights
b_h1 = tf.Variable(tf.random_normal([n_hidden_1], mean = 0, stddev=sd)) #biases
af_h1 = tf.nn.sigmoid(tf.matmul(train_inputs,w_h1) + b_h1) #activation function input

#second hidden layer
w_h2 = tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], mean = 0, stddev=sd))
b_h2 = tf.Variable(tf.random_normal([n_hidden_2], mean = 0, stddev=sd))
af_h2 = tf.nn.sigmoid(tf.matmul(af_h1,w_h2) + b_h2)

#output layer
w_out = tf.Variable(tf.random_normal([n_hidden_2,n_classes], mean = 0, stddev=sd))
b_out = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
predictions = tf.nn.softmax(tf.matmul(af_h2,w_out) + b_out)#output layer activation function 

#initialize all necessary internal vars (i.e. assign their default values)
#N.B. this needs to be done after adam optimizer is added, otherwise it gives an error
#init = tf.global_variables_initializer() #just do it during the run

#define error (cost) function and optimizer
prediction_error = -tf.reduce_sum(train_outputs * tf.log(predictions)) #cross-entropy
#prediction_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_outputs, logits=predictions)) #tf built-in cross-entropy function

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(prediction_error) #stochastic gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(prediction_error) #stochastic gradient-based optimizer with adaptive momentum estimation

#fine tune optimizer with an exponential_decay of the learning_rate
#step = tf.Variable(0, trainable=False)
#rate = tf.train.exponential_decay(learning_rate, step, 1, 0.9999)
#rate = tf.train.exponential_decay(learning_rate, global_step=step, 100000, 0.9, staircase=True)
#optimizer = tf.train.AdamOptimizer(rate).minimize(prediction_error, global_step=step)

#nn evaluation
correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(train_outputs,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
#train and test nn
#

#train neural network model
error_history = np.empty(shape=[1],dtype=float) #error history for plotting
y_true = None #for plotting and
y_pred = None #performance diagnostic

#start training
#create tf session
with tf.Session() as sess:
    #initialize internal vars (i.e. assign their default values)
    sess.run(tf.global_variables_initializer())
    #training loop of the tf nn
    for step in range(n_runs):            
        op,cost = sess.run([optimizer,prediction_error],feed_dict={train_inputs:train_features,train_outputs:train_labels})
        error_history = np.append(error_history,cost) #update error history
    print("Optimization Finished!\n")
    y_pred = sess.run(tf.argmax(predictions,1),feed_dict={train_inputs: test_features}) #predictions
    y_true = sess.run(tf.argmax(test_labels,1)) #actual
    #calculate and print training accuracy manually
    print('Training accuracy:',round(sess.run(accuracy, feed_dict={train_inputs: train_features, train_outputs: train_labels}) , 3),'\n')
    #calculate and print test accuracy manually
    print('Test accuracy:',round(sess.run(accuracy, feed_dict={train_inputs: test_features, train_outputs: test_labels}) , 3),'\n')

#plot error function - see if cost is decreasing with each step
fig = plt.figure() #figsize=(10,8)
plt.plot(error_history)
plt.axis([0,n_runs,0,np.max(error_history)])
plt.title('Error Function')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()

#confusion matrix using pandas
import pandas as pd
pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

#confusion matrix using sklearn + plotting
from sklearn.metrics import confusion_matrix
labels = ['Talker 1', 'Talker 2', 'Talker 3']
conf = confusion_matrix(y_true, y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
