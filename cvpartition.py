#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np
from random import shuffle

def cvpartition (n, k=0):
   """
   Inputs:
   n is the number of samples in the dataset.
   k is the number of folds to partition.
   Each partition has two members: training and test.
   Returns a numpy matrix of shape [n,k] and dtype=bool.
   Each column has False for samples included in the training set and True
    for the ones in the test set.
   Each partition does not overlap with the others.
   k has default value 0
   If k > n or k == 0, it will return a leave-one-out partition
   If n/k is not an integer the remaining samples will be equally but randomly 
   distributed among the partitions.

   Example:
   n      = 400
   kfolds = 10
   parts  = cvpartition(n, kfolds)
   for k in range(kfolds)
      trainset = dataset[~parts[:,k]]
      testset  = dataset[ parts[:,k]]
   """

   if k == 0:
      k = n

   if k > n:
      k = n
      print('cvpartition.py:')
      print('Samples no (' + str(n) + ') smaller than folds no (' + str(k) + '),')
      print('performing a leave-one-out.')
      #err = 'k should not be greater than x'
      #raise IOError(err)

   if k < 1:
      err = 'k should not be smaller than 1'
      raise IOError(err)

   siz   = n/k
   rem   = n%k
   sizes = np.ones(k, dtype=np.uint8) * (siz)
   if rem > 0:
      sizes[0:(rem)] += 1
      shuffle(sizes)

   kv = np.empty([n,1], dtype=np.uint16)
   x  = np.arange(0,n)
   shuffle(x)
   m  = np.zeros([n,k],dtype=bool)

   for i in range(k):
      m[x[sum(sizes[0:i]):sum(sizes[0:i+1])], i] = True

   return m

