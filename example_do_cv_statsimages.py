#!/usr/bin/python

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, subprocess, re
import numpy as np
import nibabel as nib
import pickle

#-------------------------------------------------------------------------------
def exec_comm (comm_line):
   p = subprocess.Popen(comm_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   result, err = p.communicate()
   if p.returncode != 0:
     raise IOError(err)
   return result

#-------------------------------------------------------------------------------
def find (lst, regex):
   o = []
   for i in lst:
      if re.search (regex, i):
         o.append(i)
   return o

#-------------------------------------------------------------------------------
def get_hostname ():
   import socket
   return socket.gethostname()

#-------------------------------------------------------------------------------
#fname = 'control_vs_patient_bhattacharya_99thrP_features.svmperf'
# returns 'control_vs_patient'
def get_groups_in_fname (fname):
   idx = fname.find ('_')
   idx = fname.find ('_', idx + 1)
   idx = fname.find ('_', idx + 1)
   return fname[0:idx]

#-------------------------------------------------------------------------------
def remove_ext (fname):
   return exec_comm(['remove_ext', fname]).strip()

#-------------------------------------------------------------------------------
def fslmaths (args):
   return subprocess.call('fslmaths ' + args, shell=True)
#-------------------------------------------------------------------------------

def imrm (fname):
  return exec_comm(['imrm', fname])
#-------------------------------------------------------------------------------

def imcp (orig, dest):
  return exec_comm(['imcp', orig, dest])
#-------------------------------------------------------------------------------

measures = ['jacs', 'norms', 'modulatedgm', 'trace', 'geodan', 'niftiseg_gm']
#measures = ['niftiseg_gm']

dists    = ['pearson', 'bhattacharyya', 'ttest']

studies   = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']

thrs     = [80, 90, 95, 99, 99.5, 99.9, 100]

hostname = get_hostname()

if   hostname == 'gicmed' or hostname == 'corsair': 
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari'
   rootdir    = '/media/oasis_post'
   outdir     = '/media/oasis_post'

elif hostname == 'giclus1':
   aizko_root = '/home/alexandre/work/aizkolari'
   rootdir    = '/home/alexandre/work/oasis_jesper_features'

elif hostname == 'laptosh':
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari'
   rootdir    = '/media/oasis/oasis_jesper_features'

aizko_renderstats = aizko_root + os.path.sep + 'render_statslices.py'

bakimg = rootdir + os.path.sep + 'MNI152_T1_1mm_brain.nii.gz'

nmeas    = len(measures)
ndists   = len(dists)
nstud    = len(studies)
nthrs    = len(thrs)

menum    = np.arange(nmeas,  dtype=int)
senum    = np.arange(nstud,  dtype=int)
denum    = np.arange(ndists, dtype=int)
tenum    = np.arange(nthrs,  dtype=int)

olddir = os.getcwd()

c = 0
for midx in menum:
   m = measures[midx]

   for didx in denum:
      d = dists[didx]
      cvdir   = rootdir + os.path.sep + 'cv_' + m
      cvmaskf = cvdir   + os.path.sep + 'cv_' + m

      for tidx in tenum:
         t = thrs[tidx]
         meanmaskf = cvmaskf + '_' + d  + '_' + str(t) + 'thr_meanmask'
         print (meanmaskf)

         for sidx in senum:
            i = studies[sidx]
            tstdir = cvdir + os.path.sep + d + '_' + i

            os.chdir(tstdir)

            #thresholded
            fsuff = 'thrP.nii.gz'
            lst   = find(os.listdir(tstdir), fsuff)

            fsuffx = str(t) + fsuff
            maskf = tstdir + os.path.sep + find(lst, fsuffx)[0]
            maskf = remove_ext(maskf)

            #binarize mask
            binmaskf = maskf + '.bin'
            fslmaths (maskf + ' -bin ' + binmaskf)

            if sidx == 0:
               if os.path.exists (meanmaskf + '.nii.gz'):
                  imrm(meanmaskf)

            if not os.path.exists (meanmaskf + '.nii.gz'):
               imcp(binmaskf, meanmaskf)
            else:
               fslmaths (binmaskf + ' -add ' + meanmaskf + ' ' + meanmaskf)

            imrm(binmaskf)

            os.chdir (cvdir)

         fslmaths (meanmaskf + ' -div ' + str(nstud) + ' ' + meanmaskf)

         ofstats = meanmaskf + '.slices.gif'
         exec_comm([aizko_renderstats, '--bg', bakimg, '--s1', meanmaskf, '-o', ofstats, '--s1_min', str(0.01), '--nobar'])

   os.chdir(olddir)



