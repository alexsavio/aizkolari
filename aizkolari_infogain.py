#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import scipy.stats as stats
import cmath as math
import numpy as np

from aizkolari_utils import *

#-------------------------------------------------------------------------------
def measure_infogain (datafname, labelsfile, outfname, maskfname='', exclufname=''):
   #reading label file
   labels = np.loadtxt(labelsfile, dtype=int)

   if exclufname:
      exclus = np.loadtxt(exclufname, dtype=int)

   #reading input volume
   vol = nib.load(datafname)
   n   = vol.get_shape()[3]

   if n != len(labels):
      err = 'Numbers do not match: ' + datafname + ' and ' + labelsfile
      raise IOError(err)
   elif exclufname:
      if n != len(exclus):
         err = 'Numbers do not match: ' + datafname + ' and ' + excludef
         raise IOError(err)

   au.log.debug ('Information gain of ' + os.path.basename(datafname))

   #reading volume
   data   = vol.get_data()

   #excluding subjects
   if exclufname:
      data   = data  [:,:,:,exclus == 0]
      labels = labels[exclus == 0]

   subsno = data.shape[3]

   #preprocessing data
   shape = data.shape[0:3]
   siz   = np.prod(shape)
   temp  = data.reshape(siz, subsno)
   ind   = range(len(temp))

   #calculating class entropies
   nclass = np.unique(labels)
   clentr = {} #np.zeros(len(nclass))
   #if number of classes is 2
   if nclass == 2:
      #if n of both groups is equal
      if np.sum(labels == nclass[0]) == np.sum(labels == nclass[1]):
         clentr[nclass[0]] = 1
         clentr[nclass[1]] = 1

   #if clentr has not been filled previously, calculate it with labels array
   if not clentr:
      for c in nclass:
         clentr[c] = np.sum(labels == c) / len(labels)

   if maskfname:
      mask   = nib.load(maskfname)
      mskdat = mask.get_data()
      mskdat = mskdat.reshape(siz)
      ind    = np.where(mskdat!=0)[0]

   #creating output volume file
   odat = np.zeros(shape, dtype=vol.get_data_dtype())

   for i in range(len(ind)):
      idx = ind[i]
      x = temp[idx,:]
      #p = stats.pearsonr (labels,x)[0];
      ig = 
      if math.isnan (ig): ig = 0
      odat[np.unravel_index(idx, shape)] = ig

   ovol = nib.Nifti1Image(odat, vol.get_affine())
   ovol.to_filename(outfname)


#-------------------------------------------------------------------------------
def aizkolari_data_infogain (datadir, outdir, usemask=True, excludef=''):
   olddir = os.getcwd()

   slidir = datadir + os.path.sep + slices_str()
   os.chdir(slidir)

   subjsfile  = datadir + os.path.sep + subjects_str()
   labelsfile = datadir + os.path.sep + labels_str()

   lst = os.listdir('.')
   n = count_match(lst, data_str() + '_' + slice_regex())

   for i in range(n):
      slino = zeropad(i)

      dataf = slidir + os.path.sep + data_str()    + '_' + slice_str() + '_' + slino + ext_str()
      maskf = slidir + os.path.sep + mask_str()    + '_' + slice_str() + '_' + slino + ext_str()
      outf  = outdir + os.path.sep + pearson_str() + '_' + slice_str() + '_' + slino + ext_str()
      if not os.path.isfile(dataf): 
         au.log.error ('Could not find ' + dataf)
         continue

      if not usemask:
         maskf = ''

      measure_infogain (dataf, labelsfile, outf, maskf, excludef)

   os.chdir(olddir)

