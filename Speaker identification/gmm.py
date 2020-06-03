# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:11:47 2018

@author: Shubh
"""

import numpy as np
#from sklearn import preprocessing
import mfcc
import cPickle
import os
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import warnings
warnings.filterwarnings("ignore")


dest = "speaker_models\\"

modelpath = "speaker_models\\"


def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01, 20, appendEnergy = True)
#     mfcc_feat = preprocessing.scale(mfcc_feat)
#     delta = calculate_delta(mfcc_feat)
#     combined = np.hstack((mfcc_feat,delta)) 
    return mfcc_feat


def trainModel(data_dir = "train_data"):
    speakers = os.listdir(data_dir)

    features = np.asarray(())

    for spkr_dir in speakers:
        for soundclip in os.listdir(os.path.join(data_dir, spkr_dir)):
            clip_path = os.path.abspath(os.path.join(data_dir, spkr_dir, soundclip))
            sample_rate, data = read(clip_path)

            vector = extract_features(data,sample_rate)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

        gmm = GMM(n_components = 16, n_iter = 10, covariance_type='tied', n_init = 3)
        gmm.fit(features)

        picklefile = spkr_dir+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'w'))
#         print 'Modeling completed for speaker:',picklefile," with data point = ",features.shape
        features = np.asarray(())
        
def train():
    trainModel()
    

def test(soundclip):
    gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    
    # Load the Gaussian Models
    models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
    
    clip_path = os.path.abspath(soundclip)
    sample_rate, data = read(clip_path)

    vector = extract_features(data,sample_rate)
    log_likelihood = np.zeros(len(models)) 

    for i in range(len(models)):         #checking with each model one by one
        gmm    = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    winner = np.argmax(log_likelihood)
    return speakers[winner]
            
            
            

#def testModel(wav_file):
#    gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
#    
#    # Load the Gaussian Models
#    models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
#    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
#    
#    
#    test_dir = os.path.abspath("test_data")
#    testset_size = 0
#    testset_error = 0
#    
#    for spkr_dir in os.listdir(test_dir):
#        for soundclip in os.listdir(os.path.join(test_dir, spkr_dir)):
#            clip_path = os.path.abspath(os.path.join(test_dir, spkr_dir, soundclip))
#            sample_rate, data = read(clip_path)
#    
#            vector = extract_features(data,sample_rate)
#            log_likelihood = np.zeros(len(models)) 
#    
#            for i in range(len(models)):         #checking with each model one by one
#                gmm    = models[i]
#                scores = np.array(gmm.score(vector))
#                log_likelihood[i] = scores.sum()
#            winner = np.argmax(log_likelihood)
#    
#            testset_size += 1
#            if speakers[winner] != spkr_dir:
#                testset_error += 1    
#                print "%s %s %s " % (speakers[winner], spkr_dir, u"[\u2717]")
#            else:
#                print "%s %s %s " % (speakers[winner], spkr_dir, u"[\u2713]")
#    
#    if testset_size == 0:
#        print "No test data available."
#    else:
#        print "Error on test data: %.2f%%\n" % (testset_error / testset_size * 100)
#        print "Accuracy : %.2f%%\n" % (100-(testset_error / testset_size * 100))