#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-01-15
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, sys, argparse
import numpy as np
import nibabel as nib
import scipy.io as sio

import aizkolari_utils as au
import aizkolari_export as ae

def set_parser():
   parser = argparse.ArgumentParser(description='Saves a binary file with feature sets extracted from NIFTI files.')
   parser.add_argument('-s', '--subjsf', dest='subjs', required=True,
                      help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
   parser.add_argument('-o', '--outdir', dest='outdir', required=True,
                      help='''name of the output directory where the results will be saved. \n
                      In this directory the following files will be created:
                        - included_subjects: list of full path to the subjects included in the feature set.
                        - excluded_subjects: list of full path to the subjects excluded from the feature set. if any.
                        - included_subjlabels: list of class labels of each subject in included_subjects.
                        - excluded_subjlabels: list of class labels of each subject in excluded_subjects, if any.
                        - features.*: binary file containing a NxM matrix with the features extracted from subjects (N: subj number, M: feat number).
                      ''')
   parser.add_argument('-d', '--datadir', dest='datadir', required=False,
                      help='folder path where the subjects are, if the absolute path is not included in the subjects list file.', default='')
   parser.add_argument('-m', '--featmask', default='', dest='mask', required=True,
                      help='Mask file to extract feature voxels, any voxel with values > 0 will be included in the extraction.')
   parser.add_argument('-e', '--exclude', dest='exclude', default='', required=False,
                      help='subject list mask, i.e., text file where each line has 0 or 1 indicating with 1 which subject should be excluded in the measure. To help calculating measures for cross-validation folds.')
   parser.add_argument('-t', '--type', dest='type', default=' numpybin', choices=[' numpybin',' octave',' arff', ' svmperf'], required=False,
                      help='type of the output file. Alloweds: numpybin (Numpy binary file), octave (Octave/Matlab binary file using Scipy.io.savemat), arff (Weka text file), svmperfdat (.dat for SVMPerf).')
   parser.add_argument('-n', '--name', dest='dataname', default='aizkolari_extracted', required=False,
                      help='Name of the dataset. It is used for internal usage in SVMPerf and Weka.')
   parser.add_argument('-p', '--prefix', dest='prefix', default='', required=False,
                      help='Prefix for the output filenames.')
   parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2,
                      help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

#   parser.add_argument('-p', '--thrP', dest='thresholdP', type=float, default='0.00', required=False,
#                      help='use following percentage (0-100) of ROBUST RANGE to threshold mask image (zero anything below the number).')
#   parser.add_argument('-l', '--thr', dest='threshold', type=float, default='0.00', required=False,
#                      help='use following number to threshold mask image (zero anything below the number).')
#   parser.add_argument('-a', '--abs', dest='absolute', action='store_true', required=False,
#                      help='use absolute value of mask before thresholding.')

   return parser
#-------------------------------------------------------------------------------

def set_filename (outdir, filename, otype):
   filename = outdir + os.path.sep + filename

   if   otype  == 'numpybin':
      filename += au.numpyio_ext()

   elif otype  == 'octave':
      filename += au.octaveio_ext()

   elif otype  == 'svmperf':
      filename += au.svmperfio_ext()

   elif otype  == 'arff':
      filename += au.wekaio_ext()

   else:
      err = 'Output method not recognised!'
      raise IOError(err)

   return filename

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
## START EXTRACT FEATSET
#-------------------------------------------------------------------------------
def main(argv=None):

   parser = set_parser()

   try:
      args = parser.parse_args ()
   except argparse.ArgumentError, exc:
      print (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return 0

   subjsf   = args.subjs.strip   ()
   outdir   = args.outdir.strip  ()
   datadir  = args.datadir.strip ()
   maskf    = args.mask.strip    ()
   excluf   = args.exclude.strip ()
   otype    = args.type.strip    ()
   dataname = args.dataname.strip()
   prefix   = args.prefix.strip  ()
   verbose  = args.verbosity

   au.setup_logger(verbose)

   if not os.path.exists(maskf): 
      err = 'Mask file not found: ' + maskf
      raise IOError(err)

   subjsnum = au.file_len(subjsf)

   #reading subjects list
   subjlabels = np.zeros(subjsnum, dtype=int)
   subjslist  = {}
   subjfile   = open(subjsf, 'r')
   c = 0
   for s in subjfile:
      line = s.strip().split(',')
      subjlabels[c] = int(line[0])
      subjfname = line[1].strip()
      if not os.path.isabs(subjfname) and datadir:
         subjslist[c] = datadir + os.path.sep + subjfname
      else:
         subjslist[c] = subjfname
      c += 1

   subjfile.close()

   #excluding if excluded
   if excluf:
      subjmask    = np.loadtxt(excluf, dtype=int)

      subjs       = [ subjslist[elem] for elem in subjslist  if subjmask[elem] == 0]
      labels      = subjlabels[subjmask == 0]

      exclusubjs  = [ subjslist[elem] for elem in subjslist  if subjmask[elem] == 1]
      exclulabels = subjlabels[subjmask == 1]

   else:
      subjs       = subjslist.values()
      labels      = subjlabels
      #subjmask    = np.ones(subjsnum)

   #reading mask volume
   maskd  = nib.load(maskf)
   mask   = maskd.get_data()
   nfeats = np.sum(mask > 0)
   nsubjs = len(subjs)

   #reading each subject and saving the features in a vector
   feats = np.empty([nsubjs, nfeats], dtype=float)

   #extracting features from non-excluded subjects
   c = 0
   for s in subjs:
      #check geometries
      au.check_has_same_geometry (s, maskf)

      #load data and mask it
      vold       = nib.load(s)
      vol        = vold.get_data()
      feats[c,:] = vol[mask > 0]
      c += 1

   #extracting features from excluded subjects
   if excluf:
      nexcl     = len(exclusubjs)
      exclfeats = np.empty([nexcl, nfeats], dtype=float)
      c = 0
      for s in exclusubjs:
         au.check_has_same_geometry (s, maskf)

         #load data and mask it
         vold       = nib.load(s)
         vol        = vold.get_data()
         exclfeats[c,:] = vol[mask > 0]
         c += 1

   #saving description files
   np.savetxt(outdir + os.path.sep + au.included_subjects_str(),      subjs,       fmt='%s')
   np.savetxt(outdir + os.path.sep + au.included_subjlabels_str(),    labels,      fmt='%i')

   if excluf:
      np.savetxt(outdir + os.path.sep + au.excluded_subjects_str(),   exclusubjs,  fmt='%s')
      np.savetxt(outdir + os.path.sep + au.excluded_subjlabels_str(), exclulabels, fmt='%i')

   #saving the feature matrix and labels in a binary file

   filename = set_filename (outdir, prefix + '_' + au.features_str(), otype)

   print ('Creating ' + filename)

   if otype == 'numpybin':
      np.save (filename, feats)

   elif otype == 'octave':
      sio.savemat (filename, {au.feats_str(): feats, au.labels_str(): labels})

   elif otype == 'svmperf':
      labels[labels == 0] = -1
      ae.write_svmperf_dat(filename, dataname, feats, labels)
      if excluf:
         exclulabels[exclulabels == 0] = -1
         exclfilename = set_filename(outdir, prefix + '_' + au.excluded_str() + au.feats_str(), otype)
         ae.write_svmperf_dat(exclfilename, dataname, exclfeats, exclulabels)

   elif otype == 'arff':
      featnames = np.arange(nfeats) + 1
      ae.write_arff (filename, dataname, featnames, feats, labels)

   else:
      err = 'Output method not recognised!'
      raise IOError(err)

   return filename

#-------------------------------------------------------------------------------
## END EXTRACT FEATSET
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
