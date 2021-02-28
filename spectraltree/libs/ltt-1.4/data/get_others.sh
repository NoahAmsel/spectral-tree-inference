#!/bin/sh

# 100 words of 20 newsgroups
wget http://www.cs.toronto.edu/~roweis/data/20news_w100.mat

# UCI CAR EVALUATION
wget http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
python convert.py car.data

# COIL-86
wget http://kdd.ics.uci.edu/databases/tic/ticdata2000.txt
wget http://kdd.ics.uci.edu/databases/tic/ticeval2000.txt
wget http://kdd.ics.uci.edu/databases/tic/tictgts2000.txt

# COIL-46 (from Nevin L. Zhang)
wget http://www.cs.ust.hk/faculty/lzhang/hlcmResources/coilData/coilDataTrain.txt
wget http://www.cs.ust.hk/faculty/lzhang/hlcmResources/coilData/coilDataTest.txt
grep "^[0123456789]" coilDataTrain.txt > coilDataTrain_matlab.txt
grep "^[0123456789]" coilDataTest.txt > coilDataTest_matlab.txt
