#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import re
import numpy as np
import nibabel as nib

import aizkolari_utils as au

#-------------------------------------------------------------------------------
def measure_ttest_py (mean1fname, mean2fname,\
                      var1fname, var2fname,  \
                      std1fname, std2fname,  \
                      numsubjs1, numsubjs2,  \
                      experimentname, outdir, exclude_idx=-1):

   if not os.path.exists (outdir):
      os.mkdir (outdir)

   #following the equation:
   #t = (m1 - m2) / sqrt( (v1^2)/N1 + (v2^2)/N2 )
   #from:
   #http://en.wikipedia.org/wiki/Student%27s_t-test#Unequal_sample_sizes.2C_unequal_variance

   aff   = nib.load(mean1fname).get_affine()
   mean1 = nib.load(mean1fname).get_data()
   mean2 = nib.load(mean2fname).get_data()
   var1  = nib.load( var1fname).get_data()
   var2  = nib.load( var2fname).get_data()

   ttest = (mean1 - mean2) / np.sqrt((np.square(var1) / numsubjs1) + (np.square(var2) / numsubjs2))
   ttest[np.isnan(ttest)] = 0
   ttest[np.isinf(ttest)] = 0
   #ttest = np.nan_to_num(ttest)

   ttstfname = outdir   + os.path.sep + experimentname + '_ttest' + au.ext_str()

   au.save_nibabel(ttstfname, ttest, aff)

   return ttstfname

#-------------------------------------------------------------------------------
def measure_ttest_fsl (mean1fname, mean2fname,\
                       var1fname, var2fname,  \
                       std1fname, std2fname,  \
                       numsubjs1, numsubjs2,  \
                       experimentname, outdir, exclude_idx=-1):

   if not os.path.exists (outdir):
      os.mkdir (outdir)

   olddir = os.getcwd()
   os.chdir (outdir)

   #following the equation:
   #t = (m1 - m2) / sqrt( (v1^2)/N1 + (v2^2)/N2 )
   #from:
   #http://en.wikipedia.org/wiki/Student%27s_t-test#Unequal_sample_sizes.2C_unequal_variance
   num1fname = outdir   + os.path.sep + experimentname + '_ttest_num1'
   den1fname = outdir   + os.path.sep + experimentname + '_ttest_den1'
   den2fname = outdir   + os.path.sep + experimentname + '_ttest_den2'
   denfname  = outdir   + os.path.sep + experimentname + '_ttest_denom'
   ttstfname = outdir   + os.path.sep + experimentname + '_ttest'

   au.fslmaths (mean1fname + ' -sub '       + mean2fname     + ' '       + num1fname)
   au.fslmaths (var1fname  + ' -sqr -div '  + str(numsubjs1) + ' '       + den1fname)
   au.fslmaths (var2fname  + ' -sqr -div '  + str(numsubjs2) + ' '       + den2fname)
   au.fslmaths (den1fname  + ' -add '       + den2fname      + ' -sqrt ' + denfname )
   au.fslmaths (num1fname  + ' -div '       + denfname       + ' '       + ttstfname)

#   imrm (num1fname)
#   imrm (den1fname)
#   imrm (den2fname)
#   imrm (denfname)

   os.chdir(olddir)

   return ttstfname

#-------------------------------------------------------------------------------
measure_ttest = measure_ttest_py

