#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
#import aizkolari_utils as au

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import re
import shutil
import sys
import subprocess
import commands
import warnings
import pickle
import shelve
import logging

import nibabel as nib
import numpy as np

from cvpartition import cvpartition

#----------------------------------------------------------------
def setup_logger (verbosity=1, logfname='aizkolari.log'):

  #define log level
  if verbosity == 0:
     lvl = logging.WARNING
  elif verbosity == 1:
     lvl = logging.INFO
  elif verbosity == 2:
     lvl = logging.DEBUG
  else:
     lvl = logging.WARNING

  # create logger
  global log

  log = logging.getLogger()
  log.setLevel(lvl)

  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(lvl)

  # create formatter
  formatter = logging.Formatter('%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

  # add formatter to ch
  ch.setFormatter(formatter)

  # add ch to logger
  log.addHandler(ch)

  if logfname:
    # create file handler and set level to debug
    fh = logging.FileHandler(logfname, mode='a');
    fh.setLevel(lvl)

    fh.setFormatter(formatter)
    log.addHandler(fh)

#----------------------------------------------------------------
def exec_comm (comm_line):
  p = subprocess.Popen(comm_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0:
    raise IOError(err)
  return result

#-------------------------------------------------------------------------------
def exec_comm_nowait (comm_line):
  p = subprocess.Popen(comm_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.communicate()

#-------------------------------------------------------------------------------
def file_len(fname):
  f = open(fname)
  return len(f.readlines())
#  comm_line = ['wc', '-l', fname]
#  result = exec_comm( comm_line )
#  l = result.strip()
#  l = int(l.split()[0])
#  return l

#-------------------------------------------------------------------------------
def bufcount(filename):
  f = open(filename)
  lines = 0
  buf_size = 1024 * 1024
  read_f = f.read # loop optimization

  buf = read_f(buf_size)
  while buf:
    lines += buf.count('\n')
    buf = read_f(buf_size)

  return lines

#-------------------------------------------------------------------------------
def count_match (lst, regex):
  c = 0
  for i in lst:
     if re.match (regex, i):
        c += 1
  return c

#-------------------------------------------------------------------------------
def count_match (lst, regex):
  c = 0
  for i in lst:
     if re.match (regex, i):
        c += 1
  return c

#-------------------------------------------------------------------------------
def grep_one (srch_str, filepath):
   for line in open(filepath):
      if srch_str in line:
         return line

#-------------------------------------------------------------------------------
def count_search (lst, regex):
  c = 0
  for i in lst:
     if re.search (regex, i):
        c += 1
  return c

#-------------------------------------------------------------------------------
def find (lst, regex):
  o = []
  for i in lst:
     if re.search (regex, i):
        o.append(i)
  return o

#-------------------------------------------------------------------------------
def find_name_sh (regex, wd='.', args=[]):
  olddir = os.getcwd()
  os.chdir(wd)

  comm = ['find', '-name', regex]
  if args:
    comm.extend(args)

  lst = exec_comm(comm)

  os.chdir(olddir)
  return lst

#-------------------------------------------------------------------------------
def imrm (fname):
  return exec_comm(['imrm', fname])

#-------------------------------------------------------------------------------
def imcp (orig, dest):
  return exec_comm(['imcp', orig, dest])

#-------------------------------------------------------------------------------
def immv (orig, dest):
  return exec_comm(['immv', orig, dest])

#-------------------------------------------------------------------------------
def fslstats (fname, arg):
   return exec_comm(['fslstats', fname, arg]).strip().split()

#-------------------------------------------------------------------------------
def imtest (imfname):
  return int(commands.getoutput('imtest ' + imfname))

#-------------------------------------------------------------------------------
def get_volume_intrange (fname):
   return fslstats (fname, '-R')

#-------------------------------------------------------------------------------
def add_pluses_if_exist (filepath):

   while os.path.exists(filepath):
      filepath += '+'

   return filepath

#-------------------------------------------------------------------------------
def fslslice (volume, outbase=''):
  return exec_comm(['fslslice', volume])

#-------------------------------------------------------------------------------
def fslmaths (args):
  log.debug ('Calling: fslmaths ' + args)
  return subprocess.call('fslmaths ' + args, shell=True)

#-------------------------------------------------------------------------------
def apply_mask (volfname, maskfname, outfname=''):

  volfname  = add_extension_if_needed( volfname, ext_str())
  maskfname = add_extension_if_needed(maskfname, ext_str())

  if not outfname:
    outfname = volfname

  try:
    #load data
    vol = nib.load( volfname).get_data()
    aff = nib.load( volfname).get_affine()
    msk = nib.load(maskfname).get_data()

    vol[msk == 0] = 0

    #save nifti file
    save_nibabel (outfname, vol, aff)

  except:
    log.error ("aizkolari_utils::apply_mask: Unexpected error: ", sys.exc_info()[0])
    raise


#-------------------------------------------------------------------------------
def rmdircontent (path):
  for root, dirs, files in os.walk(path):
    for f in files:
      os.unlink(os.path.join(root, f))
    for d in dirs:
      shutil.rmtree(os.path.join(root, d))

#-------------------------------------------------------------------------------
def fslval (fname, arg):
   return exec_comm(['fslval', fname, arg])

#-------------------------------------------------------------------------------
def get_extension (fpath, check_if_exists=False):
   if check_if_exists:
      if not os.path.exists (fpath):
         err = 'File not found: ' + fpath
         raise IOError(err)

   try:
      s = os.path.splitext(fpath)
      return s[-1]
   except:
      log.error( "Unexpected error: ", sys.exc_info()[0] )
      raise

#-------------------------------------------------------------------------------
def add_extension_if_needed (fpath, ext, check_if_exists=False):

   if fpath.find(ext) < 0:
     fpath += ext

   if check_if_exists:
      if not os.path.exists (fpath):
         err = 'File not found: ' + fpath
         raise IOError(err)

   return fpath

#-------------------------------------------------------------------------------
def remove_ext (fname):
   if fname.find('.nii.gz') > -1:
      return os.path.splitext(os.path.splitext(fname)[0])[0]

   return os.path.splitext(fname)[0]
   #return exec_comm(['remove_ext', fname]).strip()

#-------------------------------------------------------------------------------
def copy (source, dest):
  shutil.copy(source, dest)

#-------------------------------------------------------------------------------
def has_same_geometry (fname1, fname2):
   img1 = nib.load(fname1)
   img2 = nib.load(fname2)
   return (img1.get_shape() == img2.get_shape())

#-------------------------------------------------------------------------------
def check_has_same_geometry (fname1, fname2):
   if not has_same_geometry (fname1, fname2):
      err = 'Different shapes:' + fname1 + ' vs. ' + fname2
      raise IOError(err)

#-------------------------------------------------------------------------------
def checklist_add (checkfname, string):
   if os.path.exists(checkfname):   
      if not has_line(checkfname, string):
         append_line(checkfname, string)

#-------------------------------------------------------------------------------
def checklist_has (checkfname, string):
   has = has_line(checkfname, string)
   if has:
      log.info ('Checklist says that ' + string + ' has been done. Jumping to next step.')
   return has

#-------------------------------------------------------------------------------
def is_done (checkfname, stepname):
   if not checkfname:
      return False

   return checklist_has (checkfname, stepname)

#-------------------------------------------------------------------------------
def append_line (fname, line):
   f = open(fname,'a')
   f.write(line + '\n')
   f.close()

#-------------------------------------------------------------------------------
def has_line (fname, line):
   try:
      f = open(fname, 'r')
      for l in f:
         if (line.strip() == l.strip()):
            return True
      return False
   except:
      err = 'aizkolari_utils::has_line: Error: could not find file: ' + fname
      log.error (err)
      return False

#-------------------------------------------------------------------------------
def zeropad (x):
   return str(x).zfill(4)

#-------------------------------------------------------------------------------
def touch(fname):
   f = open(fname, 'w')
   f.write('')
   f.close()

#-------------------------------------------------------------------------------
def get_hostname ():
   import socket
   return socket.gethostname()

#-------------------------------------------------------------------------------
def get_timer():
    import time
    #deciding timer
    if sys.platform == "win32":
        # On Windows, the best timer is time.clock()
        timer = time.clock
    else:
        # On most other platforms the best timer is time.time()
        timer = time.time

    return timer


#-------------------------------------------------------------------------------
#selects in the 4th dimension only the volumes indexed by gselectfname
def group_filter (datadir, outdir, group, gselectfname, usemask=False):
   #reading input volume
   slidir   = datadir + os.path.sep + slices_str()
   outfname = outdir  + os.path.sep + group + 's'

   lst = os.listdir(slidir)
   n = count_match(lst, data_str() + '_' + slice_regex())

   try:
      select = np.loadtxt(gselectfname, dtype=int)
   except:
      log.error ("group_filter:: Unexpected error: ", sys.exc_info()[0])
      raise

   log.debug ('Merging ' + group + ' slices in ' + outdir)

   for i in range(n):
      slino = zeropad(i)

      dataf = slidir + os.path.sep + data_str()    + '_' + slice_str() + '_' + slino + ext_str()
      maskf = ''
      if usemask:
        maskf = slidir + os.path.sep + mask_str()    + '_' + slice_str() + '_' + slino + ext_str()

      outf  = outfname + '_' + slice_str() + '_' + slino + ext_str()

      n = nib.load(dataf).get_shape()[3]

      if n != len(select):
         err = 'group_filter: Numbers do not match: ' + dataf + ' and ' + gselectfname
         raise IOError(err)
      else:
         group_slice_filter (dataf, outf, select, maskf)

#-------------------------------------------------------------------------------
def write_lines (fname, lines):
   try:
      f = open(fname, 'w')
      f.writelines(lines)
      f.close()
   except IOError, err:
      log.error ('Unexpected error: ', err)
   except:
      log.error ('Unexpected error: ', str(sys.exc_info()))

#-------------------------------------------------------------------------------
def group_slice_filter (datafname, outfname, select, maskfname=''):

   if not os.path.isfile(datafname): 
      err = 'group_slice_filter: Could not find ' + datafname
      raise IOError(err)

   vol = nib.load(datafname)
   n   = vol.get_shape()[3]
   nun = np.sum(select)

   #reading volume
   data = vol.get_data()
   data = data [:,:,:,select == 1]

   if maskfname:
      mask = nib.load(maskfname).get_data()
      if   mask.ndim == 2: mask = mask[:,:, np.newaxis, np.newaxis]
      elif mask.ndim == 3: mask = mask[:,:,:, np.newaxis]

      mask = np.tile(mask, nun)
      np.putmask(data, mask == 0, 0)

   save_nibabel (outfname, data, vol.get_affine())

#-------------------------------------------------------------------------------
def read_labels_from_svmperf_file (svmperf_file):

   if not os.path.exists (svmperf_file):
      err = 'read_labels_from_svmperf_file: Could not find file ' + svmperf_file
      raise IOError (err)

   labels = []
   f = open(svmperf_file)
   for l in f:
      if l[0] != '#':
         labels.append(int(l[0:2]))

   return labels

#-------------------------------------------------------------------------------
def twofold_file_split (basefname, data, partition):
   try:
      trainf = basefname + '.traingridsearch.svmperf'
      testf  = basefname +  '.testgridsearch.svmperf'

      trainpart = partition[:,0]
      testpart  = partition[:,1]

      write_lines(trainf, data[trainpart])
      write_lines( testf, data[ testpart])

   except:
      log.error ('Unexpected error: ' + str(sys.exc_info()))
      return []

   return trainf, testf

#-------------------------------------------------------------------------------
# returns ['Accuracy', 'Precision', 'Recall', 'F1', 'PRBEP', 'ROCArea', 'AvgPrec', 'Specificity', 'Brier score']
#-------------------------------------------------------------------------------
def read_svmperf_results (logpath, predspath='', testlabels=''):

   if not os.path.exists (logpath):
      err = 'read_svmperf_results: Could not find file ' + logpath
      raise IOError (err)

   if (predspath):
      if not os.path.exists (predspath):
         err = 'read_svmperf_results: Could not find file ' + predspath
         raise IOError (err)

   results = np.zeros(9, dtype=float)

   results[0] = float(grep_one ( Accuracy_str(), logpath).strip().split(':')[1])
   results[1] = float(grep_one (Precision_str(), logpath).strip().split(':')[1])
   results[2] = float(grep_one (   Recall_str(), logpath).strip().split(':')[1])
   results[3] = float(grep_one (       F1_str(), logpath).strip().split(':')[1])
   results[4] = float(grep_one (    PRBEP_str(), logpath).strip().split(':')[1])
   results[5] = float(grep_one (  ROCArea_str(), logpath).strip().split(':')[1])
   results[6] = float(grep_one (  AvgPrec_str(), logpath).strip().split(':')[1])

   predsok = False
   if (testlabels):
      try:
         preds   = np.loadtxt(predspath, dtype = float)
         predsok = True
      except IOError:
         pass

   if (predsok):
      res = np.sign(preds)
      n   = len(testlabels)

      if n == 1:
         lbs = testlabels[0]

         tp = 0
         fp = 0
         tn = 0
         fn = 0

         if   lbs ==  1 and res ==  1 : tp = 1
         elif lbs == -1 and res ==  1 : fp = 1
         elif lbs == -1 and res == -1 : tn = 1
         elif lbs ==  1 and res == -1 : fn = 1

      else:
         lbs = np.array(testlabels)
         tp = np.sum(lbs[lbs ==  1] == res[lbs ==  1])
         fp = np.sum(lbs[lbs == -1] != res[lbs == -1])
         tn = np.sum(lbs[lbs == -1] == res[lbs == -1])
         fn = np.sum(lbs[lbs ==  1] != res[lbs ==  1])

      accuracy = (float(tp+tn)/float(n)) * 100

      if ((tp+fp) > 0): precision   = (float(tp)/float(tp+fp)) * 100
      else            : precision   = results[1]

      if ((tp+fn) > 0): recall      = (float(tp)/float(tp+fn)) * 100
      else            : recall      = results[2]

      sensitivity = recall

      if ((tn+fp) > 0) : specificity = (float(tn)/float(tn+fp)) * 100
      else             : specificity = 0

      #Brier score
      if n == 1:
         if lbs == -1: lbs = 0
         if res == -1: res = 0
      else:
         lbs[lbs == -1] = 0
         res[res == -1] = 0

      brier_score = (1/float(n)) * np.sum(np.square(res - lbs))

      results[0] = accuracy
      results[1] = precision
      results[2] = recall
      results[7] = specificity
      results[8] = brier_score

   return results

#-------------------------------------------------------------------------------
def remove_all (filelst, folder=''):
   if folder:
      try:
         for f in filelst:
            os.remove(f)
      except OSError, err:
        pass
   else:
      try:
         for f in filelst:
            os.remove(folder + os.path.sep + f)
      except OSError, err:
        pass

#-------------------------------------------------------------------------------
# SVM CALLS
#-------------------------------------------------------------------------------
def svm_linear_test (aizko_svm, trainfeatsf, testfeatsf, expname, outdir='.', cvalue=0.01, redoing=True, rocarea_opt=False, svmargs=''):
   log.debug ('Linear SVM: ' + expname + ' with C=' + str(cvalue))

   comm    = [aizko_svm, '-f', trainfeatsf, '-t', testfeatsf, '-n', expname, '-o', outdir, '-c', str(cvalue)]

   if svmargs: svmargs += ' '

   #AUC as objective function?
   if rocarea_opt:
      svmargs += '-w 3'
      comm.extend(['-l', '10'])

   if svmargs:
      comm.extend(['--svmargs', svmargs])

   if redoing:
      comm.append('--redoing')

   log.debug(' '.join(comm))

   stdout  = exec_comm(comm)
   resf    = stdout.strip().split(' ')[-1]
   results = np.loadtxt(resf)

   return results

#-------------------------------------------------------------------------------
def svm_polyrbf_test (aizko_svm, trainfeatsf, testfeatsf, expname, outdir='.', kernel=2, cvalue=0.01, pvalue=0.01, redoing=True, rocarea_opt=False, svmargs=''):

   if kernel == 1:
      pnom = 'd'
      knom = 'Polynomial'
   elif kernel == 2:
      pnom = 'g'
      knom = 'RBF'

   log.debug (knom + ' SVM: ' + expname + ' with C=' + str(cvalue) + ' and ' + pnom + '=' + str(pvalue))

   comm    = [aizko_svm, '-f', trainfeatsf, '-t', testfeatsf, '-n', expname, '-o', outdir, '--kernel', str(kernel), '-c', str(cvalue), '-p', str(pvalue)]

   if svmargs: svmargs += ' '

   if rocarea_opt:
      svmargs += '-w 3'
      comm.extend(['-l', '10'])

   if svmargs:
      comm.extend(['--svmargs', svmargs])

   if redoing:
      comm.append('--redoing')

   log.debug(' '.join(comm))

   stdout  = exec_comm(comm)
   resf    = stdout.strip().split(' ')[-1]
   results = np.loadtxt(resf)
   return results

#-------------------------------------------------------------------------------
def get_best_c_param (aizko_svm, trainfeatsf, cgrid, workdir, expname, ntimes=3, stratified=False, rocarea_opt=False, svmargs=''):

   bestc    = cgrid[0]
   rate     = 0
   nfolds   = 2
   #rate_idx = 8 #brier
   rate_idx = 0 #accuracy

   log.debug('Grid search optimization index: ' + str(rate_idx))

   if rocarea_opt:
      suffix = '.linear.rocarea.gridsearch'
   else:
      suffix = '.linear.errorrate.gridsearch'

   redoing = False

   f      = open(trainfeatsf)
   data   = f.readlines()
   nlines = len(data)
   f.close()

   if data[0][0] == '#':
      nlines -= 1
      data    = data[1:]

   data        = np.array(data)
   testlabels  = read_labels_from_svmperf_file (trainfeatsf)
   classes     = np.unique(testlabels)
   classnum    = len(classes)

   for i in np.arange(ntimes):
      #create partitions
      if not stratified:
         partition = cvpartition (nlines, nfolds)
      else:
         partition = np.empty([nlines, nfolds], dtype=int)
         for i in classes:
            gsiz    = np.sum(testlabels == i)
            gcvpart = cvpartition (gsiz, nfolds)
            for f in np.arange(nfolds):
               partition[testlabels == i,f] = gcvpart[:,f]
         partition = np.bool_(partition)

      basefname       = os.path.splitext(trainfeatsf)[:-1][0] + '.' + expname
      [trainf, testf] = twofold_file_split (basefname, data, partition)

      #evaluate best parameter
      for cval in cgrid:
         fail_count = 0
         done = False
         texpname = expname + '_c' + str(cval) + suffix
         while not done:
            try:
               results = svm_linear_test (aizko_svm, trainf, testf, texpname, workdir, cval, redoing, rocarea_opt, svmargs)
               done = True
            except:
               log.error ('Unexpected error: ' + str(sys.exc_info()))
               log.debug ('Failed. Repeating...')
               partition       = cvpartition (nlines, 2)
               basefname       = os.path.splitext(trainfeatsf)[:-1][0]
               [trainf, testf] = twofold_file_split (basefname, data, partition)
               fail_count += 1
               if fail_count < 10: pass
               else:
                   log.error ('Unexpected error: ' + str(sys.exc_info()))
                   log.debug ('Failed too many times.')
                   raise 

         if done:
            log.debug (results)
            new_rate = results[rate_idx]
            if rate < new_rate:
               rate   = new_rate
               bestc  = cval

      remove_all(find(os.listdir(workdir), suffix), workdir)

   return bestc

#-------------------------------------------------------------------------------
def get_best_polyrbf_params (aizko_svm, trainfeatsf, kernel, cgrid, paramgrid, workdir, expname, ntimes=3, stratified=False, rocarea_opt=False, svmargs=''):

   bestc    = cgrid[0]
   bestp    = paramgrid[0]
   nfolds   = 2
   rate     = 0 
   rate_idx = 8 #brier-score

   if   kernel == 1:
      suffix = '.poly'
      param  = 'd'
   elif kernel == 2:
      suffix = '.rbf'
      param  = 'g'

   if rocarea_opt:
      suffix += '.rocarea'
   else:
      suffix += '.errorrate'

   suffix += '.gridsearch'

   redoing = False

   f      = open(trainfeatsf)
   data   = f.readlines()
   nlines = len(data)
   f.close()

   if data[0][0] == '#':
      nlines -= 1
      data    = data[1:]

   data = np.array(data)

   for i in np.arange(ntimes):
      #create partitions
      if not stratified:
         partition = cvpartition (nlines, nfolds)
      else:
         partition = np.empty([nlines, nfolds], dtype=bool)
         for i in classes:
            gsiz    = np.sum(testlabels == i)
            gcvpart = cvpartition (gsiz, nfolds)
            for f in np.arange(nfolds):
               partition[testlabels == i,f] = gcvpart[:,f]
         partition = np.bool_(partition)

      basefname       = os.path.splitext(trainfeatsf)[:-1][0]
      [trainf, testf] = twofold_file_split (basefname, data, partition)

      for cval in cgrid:
         for pval in paramgrid:
            fail_count = 0
            done = False
            while not done:
               texpname = expname + '_c' + str(cval) + '_' + param + str(pval) + suffix
               try:
                  results = svm_polyrbf_test(aizko_svm, trainf, testf, texpname, workdir, cval, pval, redoing, rocarea_opt, svmargs)
                  done = True
               except:
                  log.debug ('Failed. Repeating...')
                  partition       = cvpartition (nlines, 2)
                  basefname       = os.path.splitext(trainfeatsf)[:-1][0]
                  [trainf, testf] = twofold_file_split (basefname, data, partition)
                  fail_count += 1
                  if fail_count < 10: pass
                  else:
                     log.error ('Unexpected error: ' + str(sys.exc_info()))
                     log.debug ('Failed too many times.')
                     raise 

               if done:
                 log.debug(results)
                 new_rate = results[rate_idx]
                 if rate < new_rate:
                    rate   = new_rate
                    bestc  = cval
                    bestp  = pval

      remove_all(find(os.listdir(workdir), 'gridsearch'), workdir)

   return bestc, bestp


#-------------------------------------------------------------------------------
#def get_best_rbfsvm_params_par (aizko_svm, trainfeatsf, cgrid, gammagrid, workdir, expname, ntimes=3, processes=4):

#   bestc   = 0
#   rocarea = 0

#   f      = open(trainfeatsf)
#   data   = f.readlines()
#   nlines = len(data)
#   f.close()

#   if data[0][0] == '#':
#      nlines -= 1
#      data    = data[1:]

#   data = np.array(data)

#   for i in np.arange(ntimes):
#      partition       = cvpartition (nlines, 2)
#      basefname       = os.path.splitext(trainfeatsf)[:-1][0]
#      [trainf, testf] = twofold_file_split (basefname, data, partition)

#      for cvalue in cgrid:
#         for gvalue in gammagrid:
#            texpname = expname + '_c' + str(cvalue) + '_g' + str(gvalue) + '.rocarea.gridsearch'
#            args = [aizko_linearsvm, trainf, testf, texpname, workdir, cvalue, gvalue, redoing]

##               PROCESSES = 4
##    print 'Creating pool with %d processes\n' % PROCESSES
##    pool = multiprocessing.Pool(PROCESSES)
##    print 'pool = %s' % pool
##    print

##    #
##    # Tests
##    #

##    TASKS = [(mul, (i, 7)) for i in range(10)] + \
##            [(plus, (i, 8)) for i in range(10)]

##    results = [pool.apply_async(calculate, t) for t in TASKS]
##    imap_it = pool.imap(calculatestar, TASKS)
##    imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)

#            print results
#            if rocarea < results[5]:
#               rocarea = results[5]
#               bestc   = cvalue
#               bestg   = gvalue

#      remove_all(find(os.listdir(workdir), 'gridsearch'))

#   return bestc, bestg

#-------------------------------------------------------------------------------
def rescale (data, range_min, range_max, data_min=np.NaN, data_max=np.NaN):

   if np.isnan(data_min):
      dmin = data.min()
   else:
      dmin = data_min

   if np.isnan(data_max):
      dmax = data.max()
   else:
      dmax = data_max

   try:
      d = data*((range_max-range_min)/(dmax-dmin)) + ((range_min*dmax-range_max*dmin)/(dmax-dmin))

   except:
      err = 'Rescale error.'
      raise IOError(err)

   return d, dmin, dmax

#-------------------------------------------------------------------------------
#fname = 'control_vs_patient_bhattacharya_99thrP_features.svmperf'
# returns 'control_vs_patient'
def get_groups_in_fname (fname):
   idx = fname.find ('_')
   idx = fname.find ('_', idx + 1)
   idx = fname.find ('_', idx + 1)
   return fname[0:idx]

#-------------------------------------------------------------------------------

def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n') #writeback=True

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

   #to_restore
   #my_shelf = shelve.open(filename)
   #for key in my_shelf:
   #   globals()[key]=my_shelf[key]
   #my_shelf.close()

#-------------------------------------------------------------------------------
def binarise (data, lower_bound, upper_bound, inclusive=True):
   if inclusive:
      lowers = data >= lower_bound
      uppers = data <= upper_bound
   else:
      lowers = data >  lower_bound
      uppers = data <  upper_bound

   return (lowers.astype(int) * uppers.astype(int))

#-------------------------------------------------------------------------------
#for robust limits calculation
#adapted from FSL newimage.cc code to python/numpy
#return hist,validsize
def find_histogram (vol, hist, mini, maxi, mask, use_mask):
   validsize = 0
   hist = np.zeros(hist.size, dtype=int)
   if mini == maxi:
      return -1

   fA = float(hist.size)/(maxi-mini)
   fB = (float(hist.size)*float(-mini)) / (maxi-mini)

   if use_mask:
      a = vol[mask > 0.5].flatten()
   else:
      a = vol.flatten()

   a = (a*fA + fB).astype(int)
   h = hist.size - 1

   for i in np.arange(a.size):
      hist[max(0, min(a[i], h))] += 1
      validsize += 1

   return hist, validsize

#-------------------------------------------------------------------------------
#for robust limits calculation
#adapted from FSL newimage.cc code to python/numpy
#return minval, maxval
def find_thresholds (vol, mask, use_mask=True):
   hist_bins   = 1000
   hist        = np.zeros(hist_bins, dtype=int)
   max_jumps   = 10
   top_bin     = 0
   bottom_bin  = 0
   count       = 0
   jump        = 1
   lowest_bin  = 0
   highest_bin = hist_bins-1
   validsize   = 0

   thresh98 = float(0)
   thresh2  = float(0)
   mini     = float(0)
   maxi     = float(0)

   if use_mask:
      mini = vol[mask > 0].min()
      maxi = vol[mask > 0].max()
   else:
      mini = vol.min()
      maxi = vol.max()

   while jump == 1 or ((float(thresh98) - thresh2) < (maxi - mini)/10.):
      if jump > 1:
         bottom_bin = max(bottom_bin-1, 0)
         top_bin    = min(top_bin   +1, hist_bins-1)

         tmpmin = mini + (float(bottom_bin)/float(hist_bins)) * (maxi-mini)
         maxi   = mini + (float(top_bin+1) /float(hist_bins)) * (maxi-mini)
         mini   = tmpmin

      if jump == max_jumps or mini == maxi:
         if use_mask:
            mini = vol[mask > 0].min()
            maxi = vol[mask > 0].max()
         else:
            mini = vol.min()
            maxi = vol.max()

      hist,validsize = find_histogram(vol,hist,mini,maxi,mask,use_mask);

      if validsize < 1:
         thresh2  = mini
         minval   = mini
         thresh98 = maxi
         maxval   = maxi
         return minval, maxval

      if jump == max_jumps:
         validsize   -= np.round(hist[lowest_bin]) + np.round(hist[highest_bin])
         lowest_bin  += 1
         highest_bin -= 1

      if validsize < 0:
         thresh2  = mini
         thresh98 = mini

      fA = (maxi-mini)/float(hist_bins)

      count      = 0
      bottom_bin = lowest_bin
      while count < float(validsize)/50:
         count      += np.round(hist[bottom_bin])
         bottom_bin += 1
      bottom_bin -= 1
      thresh2     = mini + float(bottom_bin) * fA

      count   = 0
      top_bin = highest_bin
      while count < float(validsize)/50:
         count   += np.round(hist[top_bin])
         top_bin -= 1
      top_bin  += 1
      thresh98  = mini + (float(top_bin) + 1) * fA

      if jump == max_jumps:
         break

      jump += 1

   minval = thresh2
   maxval = thresh98
   return minval, maxval

#-------------------------------------------------------------------------------
def robust_min (vol, mask=''):
   return find_thresholds(vol, mask)[0]

#-------------------------------------------------------------------------------
def robust_max (vol, mask=''):
   return find_thresholds(vol, mask)[1]

#-------------------------------------------------------------------------------
def threshold (data, lower_bound, upper_bound, inclusive=True):
   mask = binarise(data, lower_bound, upper_bound, inclusive)
   return data * mask

#-------------------------------------------------------------------------------
#thrP should go within [0, 100]
def threshold_robust_range (vol, thrP):
   mask       = binarise(vol, 0, vol.max()+1, False)
   limits     = find_thresholds(vol, mask)
   lowerlimit = limits[0] + float(thrP)/100*(limits[1]-limits[0])
   out        = threshold(vol, lowerlimit, vol.max()+1, True)
   return out #out.astype(vol.dtype)

#-------------------------------------------------------------------------------
def rescale (data, range_min, range_max, data_min=np.NaN, data_max=np.NaN):

   if np.isnan(data_min):
      dmin = data.min()
   else:
      dmin = data_min

   if np.isnan(data_max):
      dmax = data.max()
   else:
      dmax = data_max

   try:
      d = data*((range_max-range_min)/(dmax-dmin)) + ((range_min*dmax-range_max*dmin)/(dmax-dmin))

   except:
      err = 'Rescale error.'
      raise IOError(err)

   return d, dmin, dmax

#-------------------------------------------------------------------------------

def save_nibabel (ofname, vol, affine, header=None):
   #saves nifti file
   log.debug('Saving nifti file: ' + ofname)
   ni = nib.Nifti1Image(vol, affine, header)
   nib.save(ni, ofname)

#-------------------------------------------------------------------------------
def join_strlist_to_string (strlist):
   string = ""
   for i in strlist:
      string += " " + i
   return string

#-------------------------------------------------------------------------------

#naming functions so I get the same names for all the scripts
def abs_str():                 return 'absolute'
def bhattacharyya_str():       return 'bhattacharyya'
def checklist_str():           return 'checklist'
def data_str():                return 'data'
def exclude_str():             return 'exclude'
def excluded_str():            return 'excluded'
def ext_str():                 return '.nii.gz'
def niigz_str():               return '.nii.gz'
def fromstats_str():           return 'from statistics'
def groupfilter_str():         return 'group_filter'
def groupsizes_str():          return 'group_sizes'
def labels_str():              return 'labels'
def maskmerging_str():         return 'mask_slice_merging'
def mask_str():                return 'mask'
def mean_str():                return 'mean'
def measureperslice_str():     return 'measure_per_slice'
def measure_str():             return 'measure'
def members_str():             return 'members'
def pearson_str():             return 'pearson'
def postmerging_str():         return 'slice_merging'
def premergingdata_str():      return 'group_slice_data_merging'
def preslicingdata_str():      return 'subject_data_slicing'
def preslicingmask_str():      return 'mask_slicing'
def remove_str ():             return 'remove'
def samplesize_str ():         return 'samplesize'
def scaled_str ():             return 'scaled'
def slice_regex ():            return r"slice_[\w]*"
def slices_str():              return 'slices'
def slice_str():               return 'slice'
def stats_str():               return 'statistics'
def std_str():                 return 'stddev'
def subject_str():             return 'subject'
def subjects_str():            return 'subjects'
def temp_str():                return 'tmp'
def ttest_str():               return 'ttest'
def var_str():                 return 'var'
def sums_str():                return 'sums'
def sampsize_str():            return 'sampsize'
def size_str():                return 'size'

def subjectfiles_str():        return 'subjfiles'
def   included_subjects_str(): return 'subjfiles_included'
def   excluded_subjects_str(): return 'subjfiles_excluded'
def included_subjlabels_str(): return 'subjlabels_included'
def excluded_subjlabels_str(): return 'subjlabels_excluded'
def  features_str():           return 'features'
def  training_str():           return 'training'
def     feats_str():           return 'feats'
def    labels_str():           return 'labels'
def      test_str():           return 'test'
def   numpyio_ext():           return '.npy'
def  octaveio_ext():           return '.mat'
def svmperfio_ext():           return '.svmperf'
def    wekaio_ext():           return '.arff'

def   Accuracy_str():          return 'Accuracy'
def  Precision_str():          return 'Precision'
def  Recall_str():             return 'Recall'
def  F1_str():                 return 'F1'
def  PRBEP_str():              return 'PRBEP'
def  ROCArea_str():            return 'ROCArea'
def  AvgPrec_str():            return 'AvgPrec'

