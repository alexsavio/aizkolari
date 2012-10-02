#!/usr/bin/python

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, subprocess, re, sys
import numpy as np
import pickle

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

au.setup_logger (verbosity=2)

measures = ['jacs', 'modulatedgm', 'norms', 'trace', 'geodan']

dists    = ['pearson', 'bhattacharyya', 'ttest']

study1   = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']

#studies  = {'cv':study1, 'dd':study2}
thrs     = [80, 90, 95, 99, 99.5, 99.9, 100]

perfmeas = ['Accuracy', 'Precision', 'Recall', 'F1', 'PRBEP', 'ROCArea', 'AvgPrec', 'Specificity', 'Brier-score']

cgrid     = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
scaled    = True

#kernel
kernels   = ['linear', 'rbf']
knparams  = [1, 2]
kidx      = 0
kernel    = kernels [kidx]
nparams   = knparams[kidx]

#if you are repeating this exact same experiment, set to true
redoing = False

#set SVM optimization function to ROCArea
rocarea_opt = False

#set if stratified gridsearch is done
stratified = True

#2-fold cv grid search params
traincvnum = 3

#number of parallel processes
procs = 1

#start
hostname = au.get_hostname()

if   hostname == 'gicmed': 
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir    = '/opt/work/oasis_jesper_features/'

elif hostname == 'corsair': 
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir    = '/media/oasis_post/'

elif hostname == 'giclus1':
   aizko_root = '/home/alexandre/work/aizkolari/'
   rootdir    = '/home/alexandre/work/oasis_jesper_features/'

elif hostname == 'laptosh':
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir    = '/media/oasis/oasis_jesper_features/'

elif hostname == 'azteca':
   aizko_root = '/home/alexandre/Dropbox/Documents/phd/work/aizkolari/'
   rootdir    = '/media/oasis_post/'

aizko_svm = aizko_root + 'aizkolari_svmperf.py'

nmeas    = len(measures)
ndists   = len(dists)
nstud1   = len(study1)
nthrs    = len(thrs)
nresults = len(perfmeas)

menum    = np.arange(nmeas)
s1enum   = np.arange(nstud1)
denum    = np.arange(ndists)
tenum    = np.arange(nthrs)

residx  = {}
results = np.zeros([nmeas, ndists, nstud1, nthrs, nresults], dtype=float)
params  = np.zeros([nmeas, ndists, nstud1, nthrs], dtype=float)

olddir = os.getcwd()

rc = 0
#FULL CROSS_VALIDATION PROCESSES
for midx in menum:
   m = measures[midx]
   ofname = m

   for didx in denum:
      d = dists[didx]
      ofname += '_' + d
      cvdir   = rootdir + 'cv_' + m

      for sidx in s1enum:
         i = study1[sidx]
         tstdir  = cvdir + os.path.sep + d + '_' + i
         outdir  = tstdir
         ofname  = outdir + os.path.sep + ofname

         os.chdir(tstdir)
         au.log.debug ('cd ' + tstdir)

         for tidx in tenum:
            t = thrs[tidx]

            try:
               #in this case we will only have one trainset file per threshold
               if scaled:
                  trainfeatsf = np.sort(au.find (os.listdir('.'), str(t) + 'thrP_features.scaled.svmperf'))[0]
                  testfeatsf  = np.sort(au.find (os.listdir('.'), str(t) + 'thrP_excludedfeats.scaled.svmperf'))[0]
               else:
                  trainfeatsf = np.sort(au.find (os.listdir('.'), str(t) + 'thrP_features.svmperf'))[0]
                  testfeatsf  = np.sort(au.find (os.listdir('.'), str(t) + 'thrP_excludedfeats.svmperf'))[0]
            except:
               au.log.error ('Unexpected error: ' + str(sys.exc_info()))
               au.log.error ('Failed looking for file ' + str(t) + 'thrP*.svmperf ' + ' in ' + os.getcwd())
               exit(-1)

            expname = m + '.' + d + '.' + str(t) + 'thr'
            if not rocarea_opt:
               expname += '.errorrate'
            else:
               expname += '.rocarea'

            au.log.info ('Processing ' + i + ' ' + expname)

            prefix = ''
            if not trainfeatsf.startswith(d):
               prefix  = au.get_groups_in_fname(trainfeatsf)
               expname = prefix + '.' + expname

            trainfeatsf = tstdir + os.path.sep + trainfeatsf
            testfeatsf  = tstdir + os.path.sep + testfeatsf

            #test with grid search
            if scaled:
               texpname = expname + '.scaled.linearsvm'
            else:
               texpname = expname + '.linearsvm'

            #train grid search
            au.log.info ('Grid search')
            bestc = au.get_best_c_param (aizko_svm, trainfeatsf, cgrid, outdir, texpname, 3, stratified, rocarea_opt, '')

            params[midx,didx,sidx,tidx] = bestc

            au.log.info ('Testing ' + testfeatsf + ' with C = ' + str(bestc))
            res = au.svm_linear_test (aizko_svm, trainfeatsf, testfeatsf, texpname, outdir, bestc, redoing, rocarea_opt)

            results[midx,didx,sidx,tidx,:] = res
            rc += 1

            au.log.info('Results: ' + ' '.join(map(str,res)))

            #full test
#            for cidx in cenum:
#               cvalue   = cgrid[cidx]
#               if scaled:
#                  texpname = expname + '_C.' + str(cvalue) + '.scaled.linearsvm'
#               else:
#                  texpname = expname + '_C.' + str(cvalue) + '.linearsvm'

#               res = svm_linear_test (trainfeatsf, testfeatsf, texpname, outdir, cvalue)
#               cvresults[midx,didx,sidx,tidx,cidx,:] = res
#               rc += 1

         os.chdir(olddir)



ofsuffx = 'cv_linearsvm_cgrid_'
if not rocarea_opt:
   ofsuffx += 'l02_'
else:
   ofsuffx += 'l10_'

if scaled:
   ofsuffx += 'scaled_results'
else:
   ofsuffx += 'results'

import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
ofsuffx += now

ofsuffx += '.gridsearch'

resultsfname = 'exp_results_'    + ofsuffx + '.numpy'
paramsfname  = 'exp_parameters_' + ofsuffx + '.numpy'
indexfname   = 'exp_index_'      + ofsuffx + '.pickledump'

np.save (resultsfname, results)
np.save (paramsfname, params)

indexes = {'1:measures': measures, '2:dists': dists, '3:study1': study1, '4:thresholds': thrs, '5:cvalue': cgrid,'6:perf_measures': perfmeas}
f = open(indexfname, 'w')
pickle.dump(indexes, f, protocol=0)
f.close()
#f = open(indexfname, 'r')
#indexes = pickle.load(f)
#f.close()

print('Done')

