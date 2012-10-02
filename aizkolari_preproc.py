#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import shutil

import aizkolari_utils as au


#-------------------------------------------------------------------------------
def check_data (subsfname, labelsfname, outdir, maskfname, outbase):

   au.log.debug('Checking data')

   nsubs = au.file_len(subsfname)
   nlabs = au.file_len(labelsfname)
   if (nsubs != nlabs):
      err  = 'Not same number of lines in input files\n'
      err += labelsfname + ': ' + str(nlabs) + '\n'
      err += (subsfname   + ': ' + str(nsubs)) + '\n'
      raise IOError(err)

   if not outdir:
      outdir = os.getcwd()

   slicesdir = outdir + os.path.sep + au.slices_str()
   if not os.path.exists (slicesdir):
      os.mkdir (slicesdir)

   tempdir = outdir + os.path.sep + au.temp_str()
   if not os.path.exists (tempdir):
      os.mkdir (tempdir)

   au.log.info (slicesdir)

   #check all files
   subsfile = open(subsfname, 'r')
   if maskfname:
      for line in subsfile:
         fpath = line.strip()
         au.check_has_same_geometry (maskfname, fpath)
   else:
      path1 = ''
      path2 = ''
      for line in subsfile:
         path1 = line.strip()
         if not path2:
            path2 = line.strip()
            continue
         else:
            au.check_has_same_geometry (path1, path2)
   subsfile.close()

#-------------------------------------------------------------------------------
def slice_and_merge (subsfname, labelsfname, checkfname, outdir='', maskfname='', outbase=''):

   check_data (subsfname, labelsfname, outdir, maskfname, outbase)

   nsubs = au.file_len(subsfname)
   nlabs = au.file_len(labelsfname)

   if not outdir:
      outdir = os.getcwd()

   slicesdir = outdir + os.path.sep + au.slices_str()
   tempdir = outdir + os.path.sep + au.temp_str()
   tmpdirlst = os.listdir(tempdir)

   fpath = ''
   olddir = os.getcwd()

   if not au.is_done (checkfname, au.preslicingdata_str()):
      #slicing subjects
      au.log.info ('Slicing all subjects: takes a while')
      subsfile = open(subsfname, 'r')
      for line in subsfile:
         fpath  = line.strip()
         isfile = os.path.basename(fpath)
         osfile = tempdir + os.path.sep + isfile

         #test if file has been sliced
         fdim3 = au.fslval(fpath, 'dim3')
         regex = au.remove_ext(isfile) + '*'
         nslices = au.count_match (tmpdirlst, regex)
         if fdim3 != nslices:
            #if not, then slice it in tempdir
            au.log.debug('Slicing ' + isfile)
            os.chdir (tempdir)

            shutil.copy (fpath, tempdir)
            au.fslslice (osfile)

            os.remove(osfile)
            os.chdir (olddir)
         else:
            au.log.debug(isfile + ' previously sliced')

      subsfile.close()
      au.checklist_add (checkfname, au.preslicingdata_str())

   if not au.is_done (checkfname, au.premergingdata_str()):
      #merging each slice
      if not outbase:
         outbase = au.data_str()

      if not fpath:
         subsfile = open(subsfname, 'r')
         for line in subsfile:
            fpath = line.strip()
            break

      au.log.info ('Merging all subject slices: takes a while')

      nslices  = int(au.fslval(fpath,'dim3'))
      for slice in range(nslices):
         slicezp = au.zeropad(slice)

         mergeout  = outbase + '_slice_' + slicezp
         outdata   = slicesdir + os.path.sep + mergeout
         #check if data file exists
         if not au.imtest(outdata):
            #if not, then create it, merging all the corresponding slices
            au.log.debug('Merging slice ' + slicezp)

            imglob   = ''
            subsfile = open(subsfname, 'r')
            for line in subsfile:
               fpath   = line.strip()
               isfile  = au.remove_ext(os.path.basename(fpath)).strip()
               isfile  = tempdir + os.path.sep + isfile + '_slice_' + slicezp
               imglob += isfile + ' '

            os.system('fslmerge -t ' + outdata + ' ' + imglob)

            for f in imglob.split():
               au.imrm(f)
         else:
            au.log.debug('Slice ' + slicezp + ' previously done')

         au.checklist_add (checkfname, au.premergingdata_str())

   #slicing mask
   if maskfname:
      if os.path.exists(maskfname) and not au.is_done (checkfname, au.preslicingmask_str()):
         au.log.info('Slicing mask ' + maskfname)

         au.imcp (maskfname, slicesdir + os.path.sep + au.mask_str())
         os.chdir (slicesdir)
         au.fslslice (au.mask_str())
         au.imrm     (au.mask_str())

         au.checklist_add (checkfname, au.preslicingmask_str())
         os.chdir (olddir)

   au.log.debug('Done preprocessing')

