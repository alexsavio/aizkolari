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

#aizkod='/home/alexandre/Dropbox/Documents/phd/work/aizkolari'
#datadir='/home/alexandre/Dropbox/Documents/phd/work/aizkolari/test/data'
#classf='/home/alexandre/Dropbox/Documents/phd/work/aizkolari/test/classes'
#subjsf='/home/alexandre/Dropbox/Documents/phd/work/aizkolari/test/subslist'
#excluf='/home/alexandre/Dropbox/Documents/phd/work/aizkolari/test/exclude'
#maskf='/home/alexandre/Dropbox/Documents/phd/work/aizkolari/test/data/mask.nii.gz'
#outdir='/home/alexandre/Desktop/out'
#measure='bhattacharyya'
#measure='pearson'
#cleanup=False
#foldno=1
#parsing=False
#${aizkod}/aizkolari.py -d $datadir -c $classf -s $subjsf -o $outdir -m $maskf -f $foldno -e $excluf -n $measure

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import sys
import argparse
import numpy as np

from aizkolari_utils import *
import aizkolari_preproc as pre
import aizkolari_pearson as pear
import aizkolari_bhattacharyya as bat
import aizkolari_ttest as ttst
import aizkolari_postproc as post

#-------------------------------------------------------------------------------
def set_parser():
   parser = argparse.ArgumentParser(description='Slices and puts together a list of subjects to perform voxe-wise group calculations, e.g., Pearson correlations and bhattacharyya distance. \n The Pearson correlation is calculated between each voxel site for all subjects and the class label vector of the same subjects. \n Bhatthacharyya distance is calculated between each two groups using voxelwise Gaussian univariate distributions of each group. \n Student t-test is calculated as a Welch t-test where the two population variances are assumed to be different.')
   parser.add_argument('-c', '--classesf', dest='classes', required=True,
                      help='class label file. one line per class: <class_label>,<class_name>.')
   parser.add_argument('-s', '--subjsf', dest='subjs', required=True,
                      help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
   parser.add_argument('-o', '--outdir', dest='outdir', required=True,
                      help='name of the output directory where the results will be put.')
   parser.add_argument('-e', '--exclude', dest='exclude', default='', required=False,
                      help='subject list mask, i.e., text file where each line has 0 or 1 indicating with 1 which subject should be excluded in the measure. To help calculating measures for cross-validation folds.')
   parser.add_argument('-d', '--datadir', dest='datadir', required=False,
                      help='folder path where the subjects are, if the absolute path is not included in the subjects list file.', default='')
   parser.add_argument('-m', '--mask', dest='mask', required=False,
                      help='Mask file.')
   parser.add_argument('-n', '--measure', dest='measure', default='pearson', choices=['pearson','bhatta','bhattacharyya','ttest'], required=False,
                      help='name of the distance/correlation method. Allowed: pearson (Pearson Correlation), bhatta (Bhattacharyya distance), ttest (Student`s t-test). (default: pearson)')
   parser.add_argument('-k', '--cleanup', dest='cleanup', action='store_true',
                      help='if you want to clean up all the temp files after processing')
   parser.add_argument('-f', '--foldno', dest='foldno', required=False, type=int, default=-1,
                      help='if you want a number to identify the fold for this run, in case you will run many different folds.')
   parser.add_argument('-x', '--expname', dest='expname', required=False, type=str, default='',
                      help='if you want a name to identify this run, in case you will run many different experiments.')

   return parser
#-------------------------------------------------------------------------------

def main(argv=None):
   parsing = True

   if parsing:
      parser  = set_parser()

      try:
         args = parser.parse_args ()
      except argparse.ArgumentError, exc:
         print (exc.message + '\n' + exc.argument)
         parser.error(str(msg))
         return 0

      datadir = args.datadir.strip()
      classf  = args.classes.strip()
      subjsf  = args.subjs.strip()
      maskf   = args.mask.strip()
      outdir  = args.outdir.strip()
      excluf  = args.exclude.strip()
      measure = args.measure.strip()
      expname = args.expname.strip()
      foldno  = args.foldno
      cleanup = args.cleanup

   usemask = False
   if maskf: 
      usemask = True
   if usemask:
      if not os.path.exists(maskf):
         print ('Mask file ' + maskf + ' not found!')
         sys.exit(main())

   if foldno == -1: 
      foldnumber = ''
   else:
      foldnumber = zeropad (foldno)

   if measure == 'bhattacharyya' or measure == 'bhatta':
      measure = 'bat'
   elif measure == 'pearson':
      measure = 'pea'
   elif measure == 'ttest':
      measure = 'ttest'

   classnum = file_len(classf)
   subjsnum = file_len(subjsf)

   #reading label file
   labels     = []
   classnames = []

   labfile = open(classf, 'r')
   for l in labfile:
      line = l.strip().split(',')
      labels    .append (int(line[0]))
      classnames.append (line[1])

   labfile.close()

   #reading subjects list
   subjlabels = []
   subjs      = []
   subjfile   = open(subjsf, 'r')
   for s in subjfile:
      line = s.strip().split(',')
      subjlabels.append(int(line[0]))
      subjfname = line[1].strip()
      if not os.path.isabs(subjfname) and datadir:
         subjs.append (datadir + os.path.sep + subjfname)
      else:
         subjs.append (subjfname)

   subjfile.close()

   #if output dir does not exist, create
   if not(os.path.exists(outdir)):
      os.mkdir(outdir)

   #checklist_fname
   chkf = outdir + os.path.sep + checklist_str()
   if not(os.path.exists(chkf)):
      touch(chkf)

   #preprocessing data
   outf_subjs  = outdir + os.path.sep + subjects_str()
   outf_labels = outdir + os.path.sep + labels_str()
   np.savetxt(outf_subjs,  subjs,      fmt='%s')
   np.savetxt(outf_labels, subjlabels, fmt='%i')

   #creating folder for slices
   slidir = outdir + os.path.sep + slices_str()
   if not(os.path.exists(slidir)):
      os.mkdir(slidir)
      #slice the volumes

   #creating group and mask slices
   pre.slice_and_merge(outf_subjs, outf_labels, chkf, outdir, maskf)

   #creating measure output folder
   if measure == 'pea':
      measure_fname = pearson_str()
   elif measure == 'bat':
      measure_fname = bhattacharyya_str()
   elif measure == 'ttest':
      measure_fname = ttest_str()

   #setting the output folder mdir extension
   mdir = outdir + os.path.sep + measure_fname
   if expname:
      mdir += '_' + expname
   if foldnumber:
      mdir += '_' + foldnumber

   endstep = measure_str() + ' ' + measure_fname + ' ' + mdir
   stepdone = is_done(chkf, endstep)
   if not stepdone:
      if os.path.exists (mdir): mdir += '+'
      while os.path.exists (mdir):
         mdir += '+'
   else:
      plus = False
      while os.path.exists (mdir):
         mdir += '+'
         plus = True
      if plus:
         mdir = mdir[0:-1]

   #creating output director
   if not stepdone:
      if not os.path.exists (mdir):
         os.mkdir(mdir)

   #reading exclusion list
   outf_exclude = ''
   if (excluf):
      try:
         outf_exclude = exclude_str()
         if expname:
            outf_exclude += '_' + expname
         if foldnumber:
            outf_exclude += '_' + foldnumber
         excluded = np.loadtxt(excluf, dtype=int)
         np.savetxt(outdir + os.path.sep + outf_exclude , excluded, fmt='%i')
         np.savetxt(mdir   + os.path.sep + exclude_str(), excluded, fmt='%i')
         excluf = mdir + os.path.sep + exclude_str()

      except:
         print ('Ooops! Error processing file ' + excluf)
         print ('Unexpected error: ' + str(sys.exc_info()[0]))
         exit(1)

   #copying files to mdir
   copy(outf_subjs, mdir)
   copy(outf_labels, mdir)

   #read the measure argument and start processing
   if measure == 'pea':
      if not stepdone:

         step = measureperslice_str() + ' ' + measure_fname + ' ' + mdir
         if not is_done(chkf, step):
            pear.aizkolari_data_pearson (outdir, mdir, usemask, excluf)
            checklist_add(chkf, step)

      step = postmerging_str() + ' '+ measure_fname + ' ' + mdir
      if not is_done(chkf, step):
         pearegex = pearson_str() + '_' + slice_str() + '*'
         post.merge_slices (mdir, pearegex, pearson_str(), mdir)
         checklist_add(chkf, step)

   elif measure == 'bat' or measure == 'ttest':
      if not stepdone:
         gsize = np.zeros([len(classnames),2], dtype=int)

         for c in range(len(classnames)):
            gname  = classnames[c]
            glabel = labels    [c]
            godir  = mdir + os.path.sep + gname
            print ('Processing group ' + gname)

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

            outf_subjs  = mdir + os.path.sep + gname + '_' + subjects_str()
            outf_labels = mdir + os.path.sep + gname + '_' + labels_str()
            outf_group  = mdir + os.path.sep + gname + '_' + members_str()
            np.savetxt(outf_subjs , gsubjs,  fmt='%s')
            np.savetxt(outf_labels, glabels, fmt='%i')
            np.savetxt(outf_group , gselect, fmt='%i')

            step = groupfilter_str() + ' ' + measure_fname + ' ' + stats_str() + ' ' + gname + ' ' + mdir
            if not is_done(chkf, step):
               group_filter (outdir, mdir, gname, outf_group)
               checklist_add(chkf, step)

            step = measureperslice_str() + ' ' + measure_fname + ' ' + stats_str() + ' ' + gname + ' ' + mdir
            if not is_done(chkf, step):
               group_stats (mdir, gname)
               checklist_add(chkf, step)

            step = postmerging_str() + ' ' + measure_fname + ' ' + stats_str() + ' ' + gname + ' ' + mdir
            if not is_done(chkf, step):
               post.merge_stats_slices (mdir, gname)
               checklist_add(chkf, step)

         np.savetxt(mdir + os.path.sep + groupsizes_str(), gsize, fmt='%i,%i')

         step = postmerging_str() + ' ' + measure_fname + str(classnames) + ' ' + mdir
         if measure == 'bat':
            if not is_done(chkf, step):
               print ('Calculating Univariate Gaussian Bhattacharyya distances')
               bat.bhattacharyya_distance (mdir, classnames, chkf, foldnumber, expname)
         elif measure == 'ttest':
            if not is_done(chkf, step):
               print ('Calculating Student t-test')
               ttst.student_ttest (mdir, classnames, gsize, chkf, foldnumber, expname)

   #adding step end indication
   checklist_add(chkf, endstep)

   #merging mask slices to mdir
   step = maskmerging_str() + ' ' + measure_fname + ' ' + mdir
   if usemask and not is_done(chkf, step):
      maskregex = mask_str() + '_' + slice_str() + '*'
      post.merge_slices (slidir, maskregex, mask_str(), mdir, False)
      checklist_add(chkf, step)

   #CLEAN SPACE SUGGESTION
   rmcomm = ''
   rmcomm += 'rm -rdf ' + outdir + os.path.sep + slices_str() + ';'

   if cleanup:
      print ('Cleaning folders:')
      print (rmcomm)
      os.system (rmcomm)
   else:
      print ('If you need disk space, remove the temporary folders executing:')
      print (rmcomm.replace(';','\n'))
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())

