#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, sys, argparse
import numpy as np
import nibabel as nib

import aizkolari_utils as au

def set_parser():
   parser = argparse.ArgumentParser(description='Save thresholded NifTi volumes computed from the given volume.')
   parser.add_argument('-i', '--input', dest='input', required=True,
                      help='list of files to be thresholded. If it is a list, it must go within quotes.')
   parser.add_argument('-o', '--outdir', dest='outdir', required=False, default = '',
                      help='name of the output directory where the results will be saved. If not given, will put aside the correspondent input file.')
   parser.add_argument('-t', '--thresholds', dest='threshs', required=False, default='95'
                      help='list of floats within [0,100] separated by blank space. If it is a list, it must go within quotes')
   parser.add_argument('-m', '--mask', default='', dest='mask', required=False,
                      help='Mask file.')
   parser.add_argument('-e', '--extension', default='.nii.gz', dest='ext', required=False,
                      help='Output files extension.')
   parser.add_argument('-a', '--absolute', default=False, action='append', dest='abs', required=False,
                      help='Indicates whether to use absolute value of the volume before thresholding.')
    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=1,
                        help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

   return parser
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
## START ROBUST THRESHOLDS
#-------------------------------------------------------------------------------
def main(argv=None):

   parser = set_parser()

   try:
      args = parser.parse_args ()
   except argparse.ArgumentError, exc:
      au.log.error (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return 0

   ilst     = args.input.strip().split()
   odir     = args.outdir.strip()
   threshs  = args.threshs.strip().split()
   maskf    = args.mask.strip()
   ext      = args.ext.strip()

   au.setup_logger(args.verbosity)

   mask = nib.load(maskf).get_data()

   for i in ilst:
      im    = nib.load(i)
      ivol  = im.get_data()
      ifnom = au.remove_ext(os.path.basename(imf))

      if mask.shape != ivol.shape:
         au.log.error ("Mask file " + maskf + " and input " + i + " do not have the same shape. Skipping...")
         continue

      for t in threshs:
         au.log.info ("Thresholding " + i + " with " + str(t) + " robust lower bound.")
         out = au.threshold_robust_range (ivol, t)

         if odir:
            of = odir
         else:
            of = os.path.dirname(i)

         of += os.path.sep + ifnom + "_" + str(t) + "thrP" + ext

         om = nib.Nifti1Image(out, im.get_affine(), im.get_header(), im.extra, im.file_map)
         om.to_filename(of)

#-------------------------------------------------------------------------------
## END ROBUST THRESHOLDS
#-------------------------------------------------------------------------------

if __name__ == "__main__":
   sys.exit(main())

#mask='/home/alexandre/Desktop/oasis_jesper_features/MNI152_T1_1mm_brain_mask_dil.nii.gz'

#pthr='99.9 99.5 99 95 90 80'
#bthr='99.9 99.5 99 95 90 80'
#tthr='99.9 99.5 99 95 90 80'

##pearsons
#lstp=`find -name pearson.nii.gz`
#for i in $lstp; do
#   for t in $pthr; do
#      outdir=`dirname $i`
#      fname=`basename $i`
#      fname=`remove_ext ${fname}`
#      outfile=${outdir}/${fname}_${t}thrP

#      echo $outfile
#      fslmaths $i -mas ${mask} -abs -thrP $t ${outfile}
#   done
#done

##bhattacharyas
#lstb=`find -name *_bhattacharya.nii.gz`
#for i in $lstb; do
#   for t in $bthr; do
#      outdir=`dirname $i`
#      fname=`basename $i`
#      fname=`remove_ext ${fname}`
#      outfile=${outdir}/${fname}_${t}thrP

#      echo $outfile
#      fslmaths $i -mas ${mask} -abs -thrP $t ${outfile}
#   done
#done

##ttests
#lstt=`find -name *_ttest.nii.gz`
#for i in $lstt; do
#   for t in $tthr; do
#      outdir=`dirname $i`
#      fname=`basename $i`
#      fname=`remove_ext ${fname}`
#      outfile=${outdir}/${fname}_${t}thrP

#      echo $outfile
#      fslmaths $i -mas ${mask} -abs -thrP $t ${outfile}
#   done
#done


