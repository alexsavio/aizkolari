#!/usr/bin/python

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, subprocess, re
import numpy as np

import aizkolari_utils as au



#measures = ['jacs', 'modulatedgm', 'norms', 'trace', 'geodan']
measures = ['jacs', 'modulatedgm']
#measures = ['trace']
dists    = ['pearson', 'bhattacharyya', 'ttest']
#dists    = ['pearson']
#dists    = ['bhattacharyya']
#dists    = ['ttest']

studies  = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
#studies  = ['all']

thrs     = [80, 90, 95, 99, 99.5, 99.9, 100]
#thrs     = [100]

#studies = ['all']

hostname = get_hostname()

if   hostname == 'gicmed' or hostname == 'corsair': 
   aizko_root  = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir     = '/media/oasis_post'
   rootdatadir = '/data/oasis_jesper_features'

elif hostname == 'giclus1':
   aizko_root = '/home/alexandre/work/aizkolari/'
   rootdir    = '/home/alexandre/work/oasis_jesper_features'

elif hostname == 'laptosh':
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir    = '/media/oasis/oasis_jesper_features'

aizko_featsets = aizko_root + 'aizkolari_extract_manyfeatsets.py'
globalmask     = rootdir    + os.path.sep + 'MNI152_T1_1mm_brain_mask_dil.nii.gz'
#otypes         = ['svmperf', 'svmperf']
otypes         = ['numpybin', 'numpybin']

agesf    = rootdir + os.path.sep + 'ages'
gendersf = rootdir + os.path.sep + 'genders'

scaled = True

for m in measures:
   for s in studies:
      for d in dists:
         datadir = rootdatadir + os.path.sep + m
         cvdir   = rootdir     + os.path.sep + 'cv_' + m
         tstdir  = cvdir       + os.path.sep + d + '_' + s
         subjsf  = cvdir       + os.path.sep + 'all.txt'
         excluf  = tstdir  + os.path.sep + 'exclude'
         outdir  = tstdir

         maskregex = 'thrP.nii.gz'
         maskdir   = outdir
         lst       = os.listdir (maskdir)
         lst       = find(lst, maskregex)

         mlst = []
         for t in thrs:
            regex = str(t) + maskregex
            mlst.extend(find(lst, regex))

         maskargs  = []
         for k in mlst:
            maskpath = maskdir + os.path.sep + k
            prefix = remove_ext(os.path.basename(k))
            arg = ['-m', maskpath, '-p', prefix]
            maskargs.extend(arg)

         if s == 'all':
            otype = otypes[1]
         else:
            otype = otypes[0]

         if scaled:
            comm = [aizko_featsets, '-s', subjsf, '-o', outdir, '-d', datadir, '-g', globalmask, '-t', otype, '--scale', '--scale_min', '-1', '--scale_max', '1']
         else:
            comm = [aizko_featsets, '-s', subjsf, '-o', outdir, '-d', datadir, '-g', globalmask, '-t', otype]

         comm.extend(maskargs)

         if os.path.exists(excluf):
            comm.extend(['-e', excluf])

         exclude      = np.loadtxt(excluf,   dtype=int)
         ages         = np.loadtxt(agesf,    dtype=int)
         genders      = np.loadtxt(gendersf, dtype=str)

         gends        = np.zeros(len(genders), dtype=int)
         genders [genders == 'M'] = 1
         genders [genders == 'F'] = 0

         trainages    = ages   [exclude == 0]
         testages     = ages   [exclude == 1]
         traingenders = gends  [exclude == 0]
         testgenders  = gends  [exclude == 1]

         np.savetxt (outdir + os.path.sep + 'extra_trainset_feats_fold' + str(s) + '.txt', np.transpose(np.array([trainages,traingenders])), fmt='%i %s')
         np.savetxt (outdir + os.path.sep + 'extra_testset_feats_fold'  + str(s) + '.txt', np.transpose(np.array([ testages, testgenders])), fmt='%i %s')

         print (comm)

         #print (comm)
         proc = subprocess.call(comm)

