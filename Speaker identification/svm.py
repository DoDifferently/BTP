# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:11:21 2018

@author: Shubh
"""

from __future__ import division

import csv
import os.path
import numpy as np
import scipy.io.wavfile as wavfile
import mfcc

from sklearn.externals import joblib
from sklearn import svm

class trainModel():
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.train_file = "training_data.csv"
        
        self.gen_features()
        mfcc_list, speaker_names = self.get_tdata()
        
        # generate speaker_ids from speaker_names
        self.spkr_ntoi = {}
        self.spkr_iton = {}

        i = 0 
        for name in speaker_names:
            if name not in self.spkr_ntoi:
                self.spkr_ntoi[name] = i
                self.spkr_iton[i] = name
                i += 1
        speaker_ids = map(lambda n: self.spkr_ntoi[n], speaker_names)
        
        # train a linear svm now
        self.recognizer = svm.LinearSVC(multi_class='crammer_singer')
        self.recognizer.fit(mfcc_list, speaker_ids)

    def mfcc_to_fvec(self, ceps):
        mean = np.mean(ceps, axis=0)                               # calculate the mean 
        std = np.std(ceps, axis=0)                                 # and standard deviation of MFCC vectors 
        fvec = np.concatenate((mean, std)).tolist()                # use [mean, std] as the feature vector
        return fvec
        
    def gen_features(self):
        with open(self.train_file, 'w') as ohandle:
            melwriter = csv.writer(ohandle)
            speakers = os.listdir(self.data_dir)
            
            for spkr_dir in speakers:
                for soundclip in os.listdir(os.path.join(self.data_dir, spkr_dir)):
                    clip_path = os.path.abspath(os.path.join(self.data_dir, spkr_dir, soundclip))
                    sample_rate, data = wavfile.read(clip_path)
                    mfcc_vectors = mfcc.mfcc(data, sample_rate)
                
                    feature_vector = self.mfcc_to_fvec(mfcc_vectors)
                    feature_vector.append(spkr_dir)
                    melwriter.writerow(feature_vector)

    def get_tdata(self):
        mfcc_list = []
        speaker_names = []

        with open(self.train_file, 'r') as icsv_handle:
            melreader = csv.reader(icsv_handle)
            for row in melreader:
                mfcc_list.append(map(float, row[:-1]))
                speaker_names.append(row[-1])
        return mfcc_list, speaker_names
        
        
    def predict(self, soundclip):
        sample_rate, data = wavfile.read(os.path.abspath(soundclip))
        ceps = mfcc.mfcc(data, sample_rate)
        fvec = self.mfcc_to_fvec(ceps)
        speaker_id = self.recognizer.predict([fvec])[0]
        return self.spkr_iton[speaker_id]


def train():
    trained_model = trainModel("train_data")
    joblib.dump(trained_model, 'svm_model.pkl')

def test(soundclip):
#    trained_model = train()
    clippath = os.path.abspath(soundclip)
    trained_model = joblib.load('svm_model.pkl') 
    prediction = trained_model.predict(clippath)
    return prediction


#if __name__ == "__main__":
#    trained_model = trainModel("train_data")
#
#    test_dir = os.path.abspath("test_data")
#    testset_size = 0
#    testset_error = 0
#
#    for spkr_dir in os.listdir(test_dir):
#        for soundclip in os.listdir(os.path.join(test_dir, spkr_dir)):
#            clippath = os.path.abspath(os.path.join(test_dir, spkr_dir, soundclip))
#            prediction = trained_model.predict(clippath)
#            
#            testset_size += 1
#            if prediction != spkr_dir:
#                testset_error += 1    
#                print "%s %s %s " % (prediction, spkr_dir, u"[\u2717]")
#            else:
#                print "%s %s %s " % (prediction, spkr_dir, u"[\u2713]")
#
#    if testset_size == 0:
#        print "No test data available."
#    else:
#        print "Error on test data: %.2f%%\n" % (testset_error / testset_size * 100)
#        print "Accuracy : %.2f%%\n" % (100-(testset_error / testset_size * 100))