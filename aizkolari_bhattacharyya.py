#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import nibabel as nib
import numpy as np

import aizkolari_utils as au

#-------------------------------------------------------------------------------
#this implementation is faster around 2s than the one using FSL
def measure_bhattacharyya_distance_py (mean1fname, mean2fname, \
                                        var1fname, var2fname,  \
                                        std1fname, std2fname,  \
                                        numsubjs1, numsubjs2,  \
                                        experimentname, outdir, exclude_idx=-1):

   if not os.path.exists (outdir):
      os.mkdir (outdir)

   #following the equations:
   #1/4 * (m1-m2)^2/(var1+var2) + 1/2 * log( (var1+var2)/(2*std1*std2) )
   #from:
   #1
      #Bhattacharyya clustering with applications to mixture simplifications
      #Frank Nielsen, Sylvain Boltz, and Olivier Schwander
      #2010 International Conference on Pattern Recognition
   #2
      #The Divergence and Bhattacharyya Distance Measures in Signal Selection
      #Kailath, T.
      #http://dx.doi.org/10.1109/TCOM.1967.1089532

   aff = nib.load(mean1fname).get_affine()
   m1  = nib.load(mean1fname).get_data()
   m2  = nib.load(mean2fname).get_data()
   v1  = nib.load (var1fname).get_data()
   v2  = nib.load (var2fname).get_data()
   s1  = nib.load (std1fname).get_data()
   s2  = nib.load (std2fname).get_data()

   b1  = 0.25 * (np.square(m1 - m2) / (v1 + v2)) + 0.5  * (np.log((v1 + v2) / (2*s1*s2)))
   b1[np.isnan(b1)] = 0
   b1[np.isinf(b1)] = 0
   #b1 = np.nan_to_num(b1)

   bhatta = outdir + os.path.sep + experimentname + '_' + au.bhattacharyya_str() + au.ext_str()

   au.save_nibabel(bhatta, b1, aff)

   return bhatta

#-------------------------------------------------------------------------------
def measure_bhattacharyya_distance_fsl (mean1fname, mean2fname, \
                                        var1fname, var2fname,  \
                                        std1fname, std2fname,  \
                                        numsubjs1, numsubjs2,  \
                                        experimentname, outdir, exclude_idx=-1):

   if not os.path.exists (outdir):
      os.mkdir (outdir)

   olddir = os.getcwd()
   os.chdir (outdir)

   #following the equations:
   #1/4 * ((m1-m2)^2/(var1+var2)) + 1/2 * (log( (var1+var2)/(2*std1*std2) ))
   #from:
   #1
      #Bhattacharyya clustering with applications to mixture simplifications
      #Frank Nielsen, Sylvain Boltz, and Olivier Schwander
      #2010 International Conference on Pattern Recognition
   #2
      #The Divergence and Bhattacharyya Distance Measures in Signal Selection
      #Kailath, T.
      #http://dx.doi.org/10.1109/TCOM.1967.1089532
   num1fname = outdir   + os.path.sep + experimentname + '_bhatta_num1'
   den1fname = outdir   + os.path.sep + experimentname + '_bhatta_denom1'
   bhatta1   = outdir   + os.path.sep + experimentname + '_bhatta_first'
   au.fslmaths (mean1fname + ' -sub '    + mean2fname + ' -sqr '   + num1fname)
   au.fslmaths (var1fname  + ' -add '    + var2fname  + ' '        + den1fname)
   au.fslmaths (num1fname  + ' -div '    + den1fname  + ' -div 4 ' + bhatta1)

   num2fname = outdir  + os.path.sep + experimentname + '_bhatta_num2'
   au.fslmaths (var1fname + ' -add '    + var2fname + ' ' + num2fname)

   den2fname = outdir + os.path.sep + experimentname + '_bhatta_denom2'
   bhatta2   = outdir + os.path.sep + experimentname + '_bhatta_secnd'
   au.fslmaths (std1fname + ' -mul ' + std2fname + ' -mul 2 ' + den2fname)
   au.fslmaths (num2fname + ' -div ' + den2fname + ' -log -div 2 ' + bhatta2)

   bhatta = outdir + os.path.sep + experimentname + '_' + au.bhattacharyya_str()

   au.fslmaths (bhatta1 + ' -add ' + bhatta2 + ' ' + bhatta)

#   au.imrm (num1fname)
#   au.imrm (den1fname)
#   au.imrm (bhatta1)
#   au.imrm (num2fname)
#   au.imrm (den2fname)
#   au.imrm (bhatta2)

   os.chdir(olddir)

   return bhatta

#-------------------------------------------------------------------------------
measure_bhattacharyya_distance = measure_bhattacharyya_distance_py


