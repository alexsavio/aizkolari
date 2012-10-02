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
   parser = argparse.ArgumentParser(description='Saves a file with feature sets extracted from NIFTI files. The format of this file can be selected to be used in different software packages.')
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
   parser.add_argument('-m', '--mask', dest='mask', default=[], action='append', required=True,
                      help='Mask file to extract feature voxels, any voxel with values > 0 will be included in the extraction.')
   parser.add_argument('-g', '--globalmask', dest='globalmask', default='', required=False,
                      help='Global mask file. This mask should include all the set of voxels of the other masks. This will be used to load all the subjects in memory, otherwise you might not have enough memory.')
   parser.add_argument('-p', '--prefix', dest='prefix', default=[], action='append', required=False,
                      help='Prefix for the output filenames. You can set as many as masks, in the same order.')
   parser.add_argument('-e', '--exclude', dest='exclude', default='', required=False,
                      help='subject list mask, i.e., text file where each line has 0 or 1 indicating with 1 which subject should be excluded in the measure. To help calculating measures for cross-validation folds.')
   parser.add_argument('-t', '--type', dest='type', default='numpybin', choices=['numpybin','octave','arff', 'svmperf'], required=False,
                      help='type of the output file. Alloweds: numpybin (Numpy binary file), octave (Octave/Matlab binary file using Scipy.io.savemat), arff (Weka text file), svmperfdat (.dat for SVMPerf).')
   parser.add_argument('-n', '--name', dest='dataname', default='aizkolari_extracted', required=False,
                      help='Name of the dataset. It is used for internal usage in SVMPerf and Weka.')
   parser.add_argument('-k', '--scale', dest='scale', default=False, action='store_true', required=False,
                      help='This option will enable Range scaling of the non-excluded data and save a .range file with the max and min of the scaled dataset to scale other dataset with the same transformation.')
   parser.add_argument('-i', '--scale_min', dest='scale_min', default=-1, type=int, required=False, help='Minimum value for the new scale range.')
   parser.add_argument('-a', '--scale_max', dest='scale_max', default= 1, type=int, required=False, help='Maximum value for the new scale range.')
   parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2,
                      help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')


   return parser

#-------------------------------------------------------------------------------
def get_out_extension (otype):
   if   otype  == 'numpybin':
      ext = au.numpyio_ext()
   elif otype  == 'octave':
      ext = au.octaveio_ext()
   elif otype  == 'svmperf':
      ext = au.svmperfio_ext()
   elif otype  == 'arff':
      ext = au.wekaio_ext()
   else:
      err = 'Output method not recognised!'
      raise IOError(err)

   return ext

#-------------------------------------------------------------------------------
def get_filepath (outdir, filename, otype):

   filename = outdir + os.path.sep + filename
   try:
      filename += get_out_extension(otype)

   except:
      err = 'Output method not recognised!'
      raise IOError(err)

   return filename

#-------------------------------------------------------------------------------
def rescale (data, range_min, range_max, data_min=np.NaN, data_max=np.NaN):
   if np.isnan(data_min):
      dmin = float(data.min())
   else:
      dmin = float(data_min)

   if np.isnan(data_max):
      dmax = float(data.max())
   else:
      dmax = float(data_max)

   try:
      factor = float(((range_max-range_min)/(dmax-dmin)) + ((range_min*dmax-range_max*dmin)/(dmax-dmin)))
      d = data*factor

   except:
      err = 'Rescale error.'
      raise IOError(err)

   return d, dmin, dmax

   #kk = nib.Nifti1Image(ivol, img.get_affine(), img.get_header(), img.extra, img.file_map)
   #kk.to_filename('out5.nii.gz')

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
## START EXTRACT FEATSET
#-------------------------------------------------------------------------------
def main():

   #parsing arguments
   parser = set_parser()

   try:
      args = parser.parse_args ()
   except argparse.ArgumentError, exc:
      print (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return 0

   subjsf     = args.subjs.strip   ()
   outdir     = args.outdir.strip  ()
   datadir    = args.datadir.strip ()
   excluf     = args.exclude.strip ()
   otype      = args.type.strip    ()
   dataname   = args.dataname.strip()
   globalmask = args.globalmask.strip()

   masklst    = args.mask
   prefixes   = args.prefix
   scale      = args.scale
   scale_min  = args.scale_min
   scale_max  = args.scale_max

   verbose    = args.verbosity

   au.setup_logger(verbose)

   #checking number of files processed
   nmasks = len(masklst)
   nouts  = 0
   m = 0
   for maskf in masklst:
      if not scale:
         ofname = au.features_str() + get_out_extension(otype)
      else:
         ofname = au.features_str() + '.' + au.scaled_str() + get_out_extension(otype)

      if prefixes[m]:
         ofname = prefixes[m] + '_' + ofname

      oc     = len(au.find(os.listdir(outdir), ofname))
      nouts += oc
      m += 1

   if nouts >= nmasks:
      au.log.debug ('Nothing to do in ' + outdir + '. All files processed.')
      return -1
   else:
      au.log.debug ('Processing to output in: ' + outdir)

   #number of subjects
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

   #number of subjects
   nsubjs = len(subjs)

   #loading global mask
   if globalmask:
      gm      = nib.load(globalmask).get_data()
      subjsiz = np.sum(gm > 0)
   else:
      s       = nib.load(subjs[0])
      subjsiz = np.prod(s.shape)

   #loading subject data
   au.log.info ('Loading subject data')
   data = np.empty([nsubjs, subjsiz], dtype=float)
   c = 0
   if globalmask:
      for s in subjs:
         #load data and mask it
         print(s)
         v  = nib.load(s).get_data()
         data[c,:] = v[gm > 0]
         c += 1
   else:
      for s in subjs:
         #load data and mask it
         data[c,:] = nib.load(s).get_data().flatten()
         c += 1

   #extracting features from excluded subjects
   if excluf:
      au.log.info ('Loading excluded subject data')
      nexcl    = len(exclusubjs)
      excldata = np.empty([nexcl, subjsiz], dtype=float)
      c = 0
      if globalmask:
         for s in exclusubjs:
            #load data and mask it
            v  = nib.load(s).get_data()
            excldata[c,:] = v[gm > 0]
            c += 1
      else:
         for s in exclusubjs:
            #load data and mask it
            excldata[c,:] = nib.load(s).get_data().flatten()
            c += 1

   #for each mask in the masklst
   m = 0
   for maskf in masklst:

      #getting output prefix
      prefix = prefixes[m]
      m += 1

      #saving the feature matrix and labels in a binary file

      #setting output file name
      ofname = features_str()
      if prefix:
         ofname = prefix + '_' + ofname
      if scale:
         ofname = ofname + '.' + scaled_str()

      if excluf:
         excl_ofname = au.excluded_str() + au.feats_str()
         if prefix:
            excl_ofname = prefix + '_' + excl_ofname
         if scale:
            excl_ofname = excl_ofname + '.' + au.scaled_str()

      filename = get_filepath (outdir, ofname, otype)
      if os.path.exists(filename):
         print (filename + ' already exists. Jumping to the next.')
      else:
         print ('Creating ' + filename)

      #reading mask volume
      if not os.path.exists(maskf): 
         err = 'Mask file not found: ' + maskf
         raise IOError(err)

      print('Processing ' + maskf)

      #loading mask and masking it with globalmask
      mask   = nib.load(maskf).get_data()
      if globalmask:
         mask = mask[gm > 0]

      #number of voxels > 0 in mask
      mask   = mask.flatten()
      nfeats = np.sum(mask > 0)

      #reading each subject and saving the features in a vector
      feats = np.empty([nsubjs, nfeats], dtype=float)

      #extracting features from non-excluded subjects
      c = 0
      for s in subjs:
         #check geometries
         au.check_has_same_geometry (s, maskf)
         #mask data and save it
         feats[c,:] = data[c,mask > 0]
         c += 1

      #scaling if asked
      if scale:
         [feats, dmin, dmax] = rescale(feats, scale_min, scale_max)
         #writing in a text file the scaling values of this training set
         f = open (outdir + os.path.sep + ofname + '.scaling_range', 'w')
         f.write('#data_min, data_max, range_min, range_max')
         f.write('\n')
         f.write(str(dmin) + ',' + str(dmax) + ',' + str(scale_min) + ',' + str(scale_max))
         f.close()

      #extracting features from excluded subjects
      if excluf:
         nexcl     = len(exclusubjs)
         exclfeats = np.empty([nexcl, nfeats], dtype=float)
         c = 0
         for s in exclusubjs:
            au.check_has_same_geometry (s, maskf)

            #mask data and save it
            exclfeats[c,:] = excldata[c,mask > 0]
            c += 1

         if scale:
            [exclfeats, emin, emax] = rescale(exclfeats, scale_min, scale_max, dmin, dmax)

      #saving description files
      np.savetxt(outdir + os.path.sep + au.included_subjects_str(),      subjs,       fmt='%s')
      np.savetxt(outdir + os.path.sep + au.included_subjlabels_str(),    labels,      fmt='%i')

      if excluf:
         np.savetxt(outdir + os.path.sep + au.excluded_subjects_str(),   exclusubjs,  fmt='%s')
         np.savetxt(outdir + os.path.sep + au.excluded_subjlabels_str(), exclulabels, fmt='%i')
         exclfilename = get_filepath (outdir, excl_ofname , otype)

      #saving binary file depending on output type
      if otype == 'numpybin':
         np.save (filename, feats)

         if excluf:
            np.save (exclfilename, exclfeats)

      elif otype == 'octave':
         sio.savemat (filename, {au.feats_str(): feats, au.labels_str(): labels})
         if excluf:
            exclulabels[exclulabels == 0] = -1
            sio.savemat (exclfilename, {au.feats_str(): exclfeats, au.labels_str(): exclulabels})

      elif otype == 'svmperf':
         labels[labels == 0] = -1
         ae.write_svmperf_dat(filename, dataname, feats, labels)

         if excluf:
            exclulabels[exclulabels == 0] = -1
            ae.write_svmperf_dat(exclfilename, dataname, exclfeats, exclulabels)

      elif otype == 'arff':
         featnames = np.arange(nfeats) + 1
         ae.write_arff (filename, dataname, featnames, feats, labels)

      else:
         err = 'Output method not recognised!'
         raise IOError(err)
         return -1

   return 1

#-------------------------------------------------------------------------------
## END EXTRACT FEATSET
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
