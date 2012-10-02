#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import sys
import scipy.stats as stats
import cmath as math
import numpy as np
import nibabel as nib

import aizkolari_utils as au

#-------------------------------------------------------------------------------
def measure_pearson (datafname, labelsfile, outfname, maskfname='', exclufname='', exclude_idx=-1):
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

   exclude_log = ''
   if exclude_idx > -1:
      exclude_log = ' excluding subject ' + str(exclude_idx)

   au.log.debug ('Pearson correlation of ' + os.path.basename(datafname) + exclude_log)

   #reading volume
   data   = vol.get_data()

   #excluding subjects
   if exclufname and exclude_idx > -1:
      exclus[exclude_idx] = 1

   if exclufname:
      data   = data  [:,:,:,exclus == 0]
      labels = labels[exclus == 0]

   elif exclude_idx > -1:
      exclus = np.zeros(n, dtype=int)
      exclus[exclude_idx] = 1

      data   = data  [:,:,:,exclus == 0]
      labels = labels[exclus == 0]

   subsno = data.shape[3]

   #preprocessing data
   shape = data.shape[0:3]
   siz   = np.prod(shape)
   temp  = data.reshape(siz, subsno)
   ind   = range(len(temp))

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
      p = stats.pearsonr (labels,x)[0];

      #ldemean = labels - np.mean(labels)
      #xdemean = x - np.mean(x)
      #p = np.sum(ldemean * xdemean) / (np.sqrt(np.sum(np.square(ldemean))) * np.sqrt(np.sum(np.square(xdemean))))

      if math.isnan (p): p = 0

      odat[np.unravel_index(idx, shape)] = p

   au.save_nibabel(outfname, odat, vol.get_affine())

   return outfname


#-------------------------------------------------------------------------------
def pearson_correlation (datadir, outdir, usemask=True, excludef='', exclude_idx=-1):

   slidir = datadir + os.path.sep + au.slices_str()

   subjsfile  = datadir + os.path.sep + au.subjects_str()
   labelsfile = datadir + os.path.sep + au.labels_str()

   lst = os.listdir(slidir)
   n = au.count_match(lst, au.data_str() + '_' + au.slice_regex())

   exclude_log = ''
   if exclude_idx > -1:
      exclude_log = ' excluding subject ' + str(exclude_idx)
   
   au.log.info ('Calculating correlation of ' + slidir + os.path.sep + au.data_str() + '_' + au.slice_regex() + exclude_log)

   for i in range(n):
      slino = au.zeropad(i)

      dataf = slidir + os.path.sep + au.data_str()    + '_' + au.slice_str() + '_' + slino + au.ext_str()
      maskf = slidir + os.path.sep + au.mask_str()    + '_' + au.slice_str() + '_' + slino + au.ext_str()
      outf  = outdir + os.path.sep + au.pearson_str() + '_' + au.slice_str() + '_' + slino

      if exclude_idx > -1:
         outf += '_' + au.excluded_str() + str(exclude_idx) + au.ext_str()
      else:
         outf += au.ext_str()

      if not os.path.isfile(dataf): 
         au.log.error('Could not find ' + dataf)
         continue

      if not usemask:
         maskf = ''

      try:
         measure_pearson(dataf, labelsfile, outf, maskf, excludef, exclude_idx)
      except:
         au.log.error('pearson_correlation: Error measuring correlation on ' + dataf)
         au.log.error("Unexpected error: ", sys.exc_info()[0] )
         exit(1)

