#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#DEPENDENCIES:
#sudo apt-get install python-argparse python-numpy python-numpy-ext python-matplotlib python-scipy python-nibabel
#For development:
#sudo apt-get install ipython python-nifti python-nitime

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import sys
import argparse
import logging
import numpy as np
import nibabel as nib

import aizkolari_utils as au
import aizkolari_preproc as pre
import aizkolari_pearson as pear
import aizkolari_bhattacharyya as bat
import aizkolari_ttest as ttst
import aizkolari_postproc as post

#-------------------------------------------------------------------------------
def set_parser():
   parser = argparse.ArgumentParser(description='Slices and puts together a list of subjects to perform voxe-wise group calculations, e.g., Pearson correlations and bhattacharyya distance. \n The Pearson correlation is calculated between each voxel site for all subjects and the class label vector of the same subjects. \n Bhatthacharyya distance is calculated between each two groups using voxelwise Gaussian univariate distributions of each group. \n Student t-test is calculated as a Welch t-test where the two population variances are assumed to be different.')
   parser.add_argument('-c', '--classesf', dest='classes', required=True, help='class label file. one line per class: <class_label>,<class_name>.')
   parser.add_argument('-i', '--insubjsf', dest='subjs', required=True, help='file with a list of the volume files and its labels for the analysis. Each line: <class_label>,<subject_file>')
   parser.add_argument('-o', '--outdir', dest='outdir', required=True, help='name of the output directory where the results will be put.')
   parser.add_argument('-e', '--exclude', dest='exclude', default='', required=False, help='subject list mask, i.e., text file where each line has 0 or 1 indicating with 1 which subject should be excluded in the measure. To help calculating measures for cross-validation folds, for leave-one-out you can use the -l option.')
   parser.add_argument('-l', '--leave', dest='leave', default=-1, required=False, type=int, help='index from subject list (counting from 0) indicating one subject to be left out of the measure. For leave-one-out measures.')
   parser.add_argument('-d', '--datadir', dest='datadir', required=False, help='folder path where the subjects are, if the absolute path is not included in the subjects list file.', default='')
   parser.add_argument('-m', '--mask', dest='mask', required=False, help='Brain mask volume file for all subjects.')
   parser.add_argument('-n', '--measure', dest='measure', default='pearson', choices=['pearson','bhatta','bhattacharyya','ttest'], required=False, help='name of the distance/correlation method. Allowed: pearson (Pearson Correlation), bhatta (Bhattacharyya distance), ttest (Student`s t-test). (default: pearson)')
   parser.add_argument('-k', '--cleanup', dest='cleanup', action='store_true', help='if you want to clean up all the temp files after processing')
   parser.add_argument('-f', '--foldno', dest='foldno', required=False, type=int, default=-1, help='number to identify the fold for this run, in case you will run many different folds.')
   parser.add_argument('-x', '--expname', dest='expname', required=False, type=str, default='', help='name to identify this run, in case you will run many different experiments.')
   parser.add_argument('-a', '--absolute', dest='absolute', required=False, action='store_true', help='put this if you want absolute values of the measure.')
   parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Progression reports, 2 for Debug reports')
   parser.add_argument('--checklist', dest='checklist', required=False, action='store_true', help='If set will use and update a checklist file, which will control the steps already done in case the process is interrupted.')

   return parser

#-------------------------------------------------------------------------------
def decide_whether_usemask (maskfname):
   usemask = False
   if maskfname: 
      usemask = True
   if usemask:
      if not os.path.exists(maskfname):
         print ('Mask file ' + maskfname + ' not found!')
         usemask = False

   return usemask

#-------------------------------------------------------------------------------
def get_fold_numberstr (foldno):
   if foldno == -1: return ''
   else:            return zeropad (foldno)

#-------------------------------------------------------------------------------
def get_measure_shortname (measure_name):
   if measure_name == 'bhattacharyya' or measure_name == 'bhatta':
      measure = 'bat'
   elif measure_name == 'pearson':
      measure = 'pea'
   elif measure_name == 'ttest':
      measure = 'ttest'

   return measure

#-------------------------------------------------------------------------------
def parse_labels_file (classf):
   labels     = []
   classnames = []

   labfile = open(classf, 'r')
   for l in labfile:
      line = l.strip().split(',')
      labels    .append (int(line[0]))
      classnames.append (line[1])

   labfile.close()

   return [labels, classnames]

#-------------------------------------------------------------------------------
def parse_subjects_list (subjsfname, datadir=''):
   subjlabels = []
   subjs      = []

   if datadir:
      datadir += os.path.sep

   try:
      subjfile   = open(subjsfname, 'r')
      for s in subjfile:
         line = s.strip().split(',')
         subjlabels.append(int(line[0]))
         subjfname = line[1].strip()
         if not os.path.isabs(subjfname):
            subjs.append (datadir + subjfname)
         else:
            subjs.append (subjfname)

      subjfile.close()
   except:
      log.error( "Unexpected error: ", sys.exc_info()[0] )
      sys.exit(-1)

   return [subjlabels, subjs]

#-------------------------------------------------------------------------------
def parse_exclude_list (excluf, leave=-1):
   excluded =[]

   if (excluf):
      try:
         excluded = np.loadtxt(excluf, dtype=int)

         #if leave already excluded, dont take it into account
         if leave > -1:
            if excluded[leave] == 1:
                au.log.warn ('Your left out subject (-l) is already being excluded in the exclusion file (-e).')
                leave = -1

      except:
         au.log.error ('Error processing file ' + excluf)
         au.log.error ('Unexpected error: ' + str(sys.exc_info()[0]))
         sys.exit(-1)

   return excluded

#-------------------------------------------------------------------------------
def main(argv=None):

   parser  = set_parser()

   try:
      args = parser.parse_args ()
   except argparse.ArgumentError, exc:
      print (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return -1

   datadir  = args.datadir.strip()
   classf   = args.classes.strip()
   subjsf   = args.subjs.strip()
   maskf    = args.mask.strip()
   outdir   = args.outdir.strip()
   excluf   = args.exclude.strip()
   measure  = args.measure.strip()
   expname  = args.expname.strip()
   foldno   = args.foldno
   cleanup  = args.cleanup
   leave    = args.leave
   absval   = args.absolute
   verbose  = args.verbosity
   chklst   = args.checklist

   au.setup_logger(verbose)

   usemask = decide_whether_usemask(maskf)

   foldno = get_fold_numberstr (foldno)

   measure = get_measure_shortname (measure)

   classnum = au.file_len(classf)
   subjsnum = au.file_len(subjsf)

   #reading label file
   [labels, classnames] = parse_labels_file (classf)

   #reading subjects list
   [subjlabels, subjs] = parse_subjects_list (subjsf, datadir)

   #if output dir does not exist, create
   if not(os.path.exists(outdir)):
      os.mkdir(outdir)

   #checklist_fname
   if chklst:
      chkf = outdir + os.path.sep + au.checklist_str()
      if not(os.path.exists(chkf)):
         au.touch(chkf)
   else:
      chkf = ''

   #saving data in files where further processes can find them
   outf_subjs  = outdir + os.path.sep + au.subjects_str()
   outf_labels = outdir + os.path.sep + au.labels_str()
   np.savetxt(outf_subjs,  subjs,      fmt='%s')
   np.savetxt(outf_labels, subjlabels, fmt='%i')

   #creating folder for slices
   slidir = outdir + os.path.sep + au.slices_str()
   if not(os.path.exists(slidir)):
      os.mkdir(slidir)
      #slice the volumes

   #creating group and mask slices
   pre.slice_and_merge(outf_subjs, outf_labels, chkf, outdir, maskf)

   #creating measure output folder
   if measure == 'pea':
      measure_fname = au.pearson_str()
   elif measure == 'bat':
      measure_fname = au.bhattacharyya_str()
   elif measure == 'ttest':
      measure_fname = au.ttest_str()

   #checking the leave parameter
   if leave > (subjsnum - 1):
      au.log.warning('aizkolari_measure: the leave (-l) argument value is ' + str(leave) + ', bigger than the last index of subject: ' + str(subjsnum - 1) + '. Im setting it to -1.')
      leave = -1

   #reading exclusion list
   excluded = parse_exclude_list (excluf, leave)

   #setting the output folder mdir extension
   mdir = outdir + os.path.sep + measure_fname
   if expname:
      mdir += '_' + expname
   if foldno:
      mdir += '_' + foldno

   #setting the stats folder
   statsdir = outdir + os.path.sep + au.stats_str()
   if expname:
      statsdir += '_' + expname
   if foldno:
      statsdir += '_' + foldno


   #setting a string with step parameters
   step_params = ' ' + measure_fname + ' ' + mdir

   absolute_str = ''
   if absval:
      absolute_str = ' ' + au.abs_str()
   step_params += absolute_str

   leave_str = ''
   if leave > -1:
      leave_str = ' excluding subject ' + str(leave)
   step_params += leave_str

   #checking if this measure has already been done
   endstep = au.measure_str() + step_params

   stepdone = au.is_done(chkf, endstep)
   #add pluses to output dir if it already exists
   if stepdone:
      while os.path.exists (mdir):
         mdir += '+'
   else: #work in the last folder used
      plus = False
      while os.path.exists (mdir):
         mdir += '+'
         plus = True
      if plus:
         mdir = mdir[0:-1]

   #setting statsdir
   pluses = mdir.count('+')
   for i in np.arange(pluses): statsdir += '+'

   #merging mask slices to mdir
   if not stepdone:
      #creating output folders
      if not os.path.exists (mdir):
         os.mkdir(mdir)

      #copying files to mdir
      au.copy(outf_subjs, mdir)
      au.copy(outf_labels, mdir)

      #saving exclude files in mdir
      outf_exclude = ''
      if (excluf):
         outf_exclude = au.exclude_str()
         if expname:
            outf_exclude += '_' + expname
         if foldnumber:
            outf_exclude += '_' + foldnumber

         np.savetxt(outdir + os.path.sep + outf_exclude , excluded, fmt='%i')
         np.savetxt(mdir   + os.path.sep + au.exclude_str(), excluded, fmt='%i')
         excluf = mdir + os.path.sep + au.exclude_str()

      step = au.maskmerging_str() + ' ' + measure_fname + ' ' + mdir
      if usemask and not au.is_done(chkf, step):
         maskregex = au.mask_str() + '_' + au.slice_str() + '*'
         post.merge_slices (slidir, maskregex, au.mask_str(), mdir, False)
         au.checklist_add(chkf, step)

      #CORRELATION
      #read the measure argument and start processing
      if measure == 'pea':
         #measure pearson correlation for each population slice
         step = au.measureperslice_str() + step_params
         if not au.is_done(chkf, step):
            pear.pearson_correlation (outdir, mdir, usemask, excluf, leave)
            au.checklist_add(chkf, step)

         #merge all correlation slice measures
         step = au.postmerging_str() + step_params
         if not au.is_done(chkf, step):
            pearegex = au.pearson_str() + '_' + au.slice_str() + '*'
            peameasf = mdir + os.path.sep + au.pearson_str()

            if leave > -1:
               pearegex += '_' + au.excluded_str() + str(leave) + '*'
               peameasf += '_' + au.excluded_str() + str(leave) + '_' + au.pearson_str()

            post.merge_slices (mdir, pearegex, peameasf, mdir)

            if absval:
               post.change_to_absolute_values(peameasf)

            au.checklist_add(chkf, step)

      #BHATTACHARYYA AND T-TEST
      elif measure == 'bat' or measure == 'ttest':

         if not os.path.exists (statsdir):
            os.mkdir(statsdir)

         gsize = np.zeros([len(classnames),2], dtype=int)

         for c in range(len(classnames)):
            gname  = classnames[c]
            glabel = labels    [c]
            godir  = mdir + os.path.sep + gname
            au.log.debug ('Processing group ' + gname)

            gselect = np.zeros(len(subjs))
            gsubjs  = list()
            glabels = list()
            for s in range(len(subjs)):
               slabel = subjlabels[s]
               if slabel == glabel:
                  gsubjs .append (subjs[s])
                  glabels.append (slabel)
                  gselect[s] = 1
                  if outf_exclude:
                     if excluded[s]:
                        gselect[s] = 0

            gsize[c,0] = glabel
            gsize[c,1] = np.sum(gselect)

            outf_subjs  = mdir + os.path.sep + gname + '_' + au.subjects_str()
            outf_labels = mdir + os.path.sep + gname + '_' + au.labels_str()
            outf_group  = mdir + os.path.sep + gname + '_' + au.members_str()
            np.savetxt(outf_subjs , gsubjs,  fmt='%s')
            np.savetxt(outf_labels, glabels, fmt='%i')
            np.savetxt(outf_group , gselect, fmt='%i')

            step = au.groupfilter_str() + ' ' + gname + ' ' + statsdir
            if not au.is_done(chkf, step):
               au.group_filter (outdir, statsdir, gname, outf_group, usemask)
               au.checklist_add(chkf, step)

            grp_step_params = ' ' + au.stats_str() + ' ' + gname + ' ' + statsdir
            step = au.measureperslice_str() + grp_step_params
            if not au.is_done(chkf, step):
               post.group_stats (statsdir, gname, gsize[c,1], statsdir)
               au.checklist_add(chkf, step)

            statfnames = {}
            step = au.postmerging_str() + grp_step_params
            if not au.is_done(chkf, step):
               statfnames[gname] = post.merge_stats_slices (statsdir, gname)
               au.checklist_add(chkf, step)

         sampsizef = mdir + os.path.sep + au.groupsizes_str()
         np.savetxt(sampsizef, gsize, fmt='%i,%i')

         #decide which group distance function to use
         if measure == 'bat':
             distance_func = bat.measure_bhattacharyya_distance
         elif measure == 'ttest':
             distance_func = ttst.measure_ttest

         #now we deal with the indexed excluded subject
         step = au.postmerging_str() + ' ' + str(classnames) + step_params
         exsubf = ''
         exclas = ''

         if leave > -1:
            exsubf = subjs[leave]
            exclas = classnames[subjlabels[leave]]

         if not au.is_done(chkf, step):
            #group distance called here, in charge of removing the 'leave' subject from stats as well
            measfname = post.group_distance (distance_func, statsdir, classnames, gsize, chkf, absval, mdir, foldno, expname, leave, exsubf, exclas)

            if usemask:
               au.apply_mask (measfname, mdir + os.path.sep + au.mask_str())

            au.checklist_add(chkf, step)

      #adding step end indication
      au.checklist_add(chkf, endstep)

      #CLEAN SPACE SUGGESTION
      rmcomm = 'rm -rf ' + outdir + os.path.sep + au.slices_str() + ';'

      if cleanup:
         au.log.debug ('Cleaning folders:')
         au.log.info  (rmcomm)
         os.system (rmcomm)
      else:
         au.log.info ('If you need disk space, remove the temporary folders executing:')
         au.log.info (rmcomm.replace(';','\n'))

         if leave > -1:
            au.log.info ('You should not remove these files if you are doing further leave-one-out measures.')

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

