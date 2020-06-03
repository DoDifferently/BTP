# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:10:48 2018

@author: Shubh
"""

import numpy
import base
from scipy.fftpack import dct

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy:
        feat[:,0] = numpy.log(energy)
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    highfreq= highfreq or samplerate/2
    signal = base.preemphasis(signal,preemph)
    frames = base.framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = base.powspec(frames,nfft)
    energy = numpy.sum(pspec,1)                                     # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log
    fb = get_filterbanks(nfilt,nfft,samplerate)
    feat = numpy.dot(pspec,fb.T)                                    # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat)       # if feat is zero, we get problems with log
    return feat,energy

def hz2mel(hz):
    return 2595 * numpy.log10(1+hz/700.0)
    
def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    highfreq= highfreq or samplerate/2
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)
#     print nfilt,nfft/2+1
    fbank = numpy.zeros([nfilt,int(nfft/2+1)])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank                 
    
def lifter(cepstra,L=22):
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1+ (L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra