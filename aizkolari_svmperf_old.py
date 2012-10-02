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

import os
import sys
import argparse
import subprocess
import logging

import numpy as np
import scipy.io as sio

import aizkolari_utils as au

def set_parser():
   parser = argparse.ArgumentParser(description='''Returns classification results with SVMPerf. \n
        You should have SVMPerf installed in your system and accessible from the command line as svmperf_learn and svmperf_classifiy, or you can specify their directory with the -a parameter.\n
     ''')
   parser.add_argument('-f', '--feats', dest='feats', required=True, 
      help='SVMPerf formatted text features file used for training. Declare as many of these as you want.')
   parser.add_argument('-t', '--test', dest='testf', required=True, 
      help='SVMPerf formatted text features file used for test. As many as training sets.')
   parser.add_argument('-o', '--outdir', dest='outdir', required=True, 
      help='Output directory path.')
   parser.add_argument('-a', '--svmdir', default='', dest='svmdir', required=False, help='Directory where svm_perf_classify and svm_perf_learn are. Use this if they are not in your terminal execution path')
   parser.add_argument('-n', '--name', dest='expname', default='', required=False, 
      help='Experiment name will be put as suffix in the output file names. Otherwise, a name extracted from the features files will be used.')
   parser.add_argument('-c', '--regularizer', dest='cvalue', default=0.01, type=float, required=False, help='Value for the SVM C regularizer.')
   parser.add_argument('-l', '--loss_function', dest='lossf', default=2, type=int, required=False, help='Index to indicate the loss function of the classifier: 0: Zero/one loss, 1: F1, 2: Errorrate, 3: Prec/Rec Breakeven, 4:  Prec@k, 5: Rec@k, 10: ROCArea. See SVMPerf documentation for further details')
   parser.add_argument('-g', '--svmargs', dest='svmargs', default='', type=str, required=False, help='Other arguments for SVMPerf learn.')
   parser.add_argument('-r', '--redoing', dest='redoing', default=False, action='store_true', required=False, help='Allows the script to read the results file if it already exists.')
   parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2,
                      help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

   return parser
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
def svmlin_learn (binpath, trainset_fname, outmodel_fname, log_fname, cvalue=1, lossf=2, svmargs='', expname=''):

   logging.basicConfig(filename=log_fname, filemode='w', format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)

   logging.info('#-------------------------------------------------')
   logging.info('\n')
   logging.info('#TRAINING:'  + expname)
   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')

   logfd = open(log_fname, mode)
   logfd.flush()

   comm = [binpath, '-l', str(lossf), '-c', str(cvalue)]

   if svmargs:
      comm.extend(svmargs.split(' '))

   comm.extend([trainset_fname, outmodel_fname])

   #print(comm)

   debug_here()

   subprocess.call (comm, stdout=logfd)

   logfd.close()
   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')
   logging.info('#END TRAINING: ' + expname)
   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')

#-------------------------------------------------------------------------------
def svmlin_test (binpath, model_fname, testset_fname, results_fname, log_fname, expname=''):

   mode = 'a'
   if not os.path.exists (log_fname):
      mode = 'w'

   logging.basicConfig(filename=log_fname, filemode=mode, format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)

   logging.info('#-------------------------------------------------')
   logging.info('\n')
   logging.info('#CLASSIFYING:'  + expname)
   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')
   logfd.flush()

   comm = [binpath, testset_fname, model_fname, results_fname]
   subprocess.call (comm, stdout=logfd)

   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')
   logging.info('#END TRAINING: ' + expname)
   logging.info('\n')
   logging.info('#-------------------------------------------------')
   logging.info('\n')

   logfd.close()


#-------------------------------------------------------------------------------
## MAIN
#-------------------------------------------------------------------------------
def main ():

   parser = set_parser()

   try:
      args = parser.parse_args  ()
   except argparse.ArgumentError, exc:
      au.log.error (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return 0

   outdir  = args.outdir.strip  ()
   svmdir  = args.svmdir.strip  ()
   expname = args.expname.strip ()
   featsf  = args.feats.strip()
   testf   = args.testf.strip()
   cvalue  = args.cvalue
   lossf   = args.lossf
   svmargs = args.svmargs.strip()
   redoing = args.redoing
   verbose = args.verbosity

   au.setup_logger(verbose)

   #setting command paths
   svm_class = 'svm_perf_classify'
   svm_learn = 'svm_perf_learn'
   if svmdir:
      svm_class = svmdir + os.path.sep + svm_class
      svm_learn = svmdir + os.path.sep + svm_learn

   train_fname = featsf
   test_fname  = testf

   #creating output filename
   if expname:
      ofname = os.path.basename(expname)
   else:
      ofname = au.remove_ext(os.path.basename(train_fname))

   modelf = outdir + os.path.sep + ofname + '.model.dat'
   predsf = outdir + os.path.sep + ofname + '.preds.txt'
   logf   = outdir + os.path.sep + ofname + '.log.txt'
   resf   = outdir + os.path.sep + ofname + '.results.txt'

   if redoing:
      if os.path.exists(resf):
         au.log.info('The results already are in ' + resf)
         return 0

   modelf = au.add_pluses_if_exist (modelf)
   predsf = au.add_pluses_if_exist (predsf)
   logf   = au.add_pluses_if_exist (logf)

   testlabels = au.read_labels_from_svmperf_file (testf)

   #non-linear kernel
   #learn_opts = ['-c 1 -t 2 -g 1 -w 3 -l 10 --i 0 --t 2 --k 500 --b 0'];
   #learn_opts = ['-c 1 -t 2 -w 9 -l 2  --i 2 --k 500 --b 0'];
   svmlin_learn (svm_learn, train_fname, modelf, logf, cvalue, lossf, svmargs)

   svmlin_test (svm_class, modelf, test_fname, predsf, logf)

   results = au.read_svmperf_results (logf, predsf, testlabels)

   np.savetxt (resf, results, fmt='%f')

   print ('The results are in ' + logf + ' and also in the same order in ' + resf)


#-------------------------------------------------------------------------------
## END MAIN
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
