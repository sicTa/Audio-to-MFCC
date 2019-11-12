# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:44:28 2019

@author: sicTa
"""

import AudioAnalyzer
import numpy as np

aa = AudioAnalyzer.AudioAnalyzer()
mfcc = aa.MFCC('OSR_us_000_0061_8k.wav')
aa.plot(mfcc)


#using mfcc's we can make a simple list 

x = aa.feature_vector2('OSR_us_000_0061_8k.wav')


#print(mfcc[100])
print(x[100])