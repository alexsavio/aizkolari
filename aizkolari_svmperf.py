#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-01-15
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

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
    parser.add_argument('-f', '--featsf', dest='feats', required=True, help='SVMPerf formatted text features file used for training. Declare as many of these as you want.')
    parser.add_argument('-t', '--testf', dest='testf', required=True, help='SVMPerf formatted text features file used for test. As many as training sets.')
    parser.add_argument('-o', '--outdir', dest='outdir', required=True, help='Output directory path.')
    parser.add_argument('-d', '--svmdir', dest='svmdir', default='', required=False, help='Directory where svm_perf_classify and svm_perf_learn are. Use this if they are not in your PATH')
    parser.add_argument('-k', '--kernel', dest='kernel', default=0, choices=[0, 1, 2], type=int, required=False, help='type of kernel function: 0: linear, 1: polynomial or 2: rbf (radial basis function).')
    parser.add_argument('-n', '--name', dest='expname', default='', required=False, help='Experiment name will be put as suffix in the output file names. Otherwise, a name extracted from the features files will be used.')
    parser.add_argument('-c', '--regularizer', dest='cvalue', default=0.01, type=float, required=False, help='Value for the SVM C regularizer.')
    parser.add_argument('-p', '--nlparam', dest='nlparam', default=0.01, type=float, required=False, help='Value for the non-linear parameters: gamma in case of RBF kernel, exponent in case of polynomial.')
    parser.add_argument('-l', '--loss_func', dest='lossf', default=2, type=int, required=False, help='Index to indicate the loss function of the classifier: 0: Zero/one loss, 1: F1, 2: Errorrate, 3: Prec/Rec Breakeven, 4:  Prec@k, 5: Rec@k, 10: ROCArea. See SVMPerf documentation for further details')
    parser.add_argument('-g', '--svmargs', dest='svmargs', default='', type=str, required=False, help='Other arguments for SVMPerf learn. Be careful with not setting any other parameter already set with these parameters. svmargs will be passed to the SVM grid search and to the final test.')
    parser.add_argument('-r', '--redoing', dest='redoing', default=False, action='store_true', required=False, help='Allows the script to read the results file if it already exists.')
    parser.add_argument('-s', '--gridsearch', dest='gridsearch', default=False, action='store_true', required=False, help='This option will enable grid search for SVM parameters using a stratified 3x2-fold cross-test with the training set using the parameter values defined for search. The parameters with best fold-average Brier score will be chosen for validation. Available for linear, polynomial or RBF kernels and Errorrate or ROCArea loss functions only.')
    parser.add_argument('--reggrid', dest='reggrid', default='0.001, 0.01, 0.1, 1, 10, 100', required=False, help='List of values for the C regularizer to be used in the gridsearch.')
    parser.add_argument('--pargrid', dest='pargrid', default='0.001, 0.01, 0.1, 1, 10, 100', required=False, help='List of values for the second SVM parameter (if not linear) to be used in the gridsearch. For RBF and polynomial kernels only.')
    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Progression reports, 2 for Debug reports')

    return parser

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
def svm_learn (binpath, trainset_fname, outmodel_fname, log_fname, cvalue=1, lossf=2, svmargs='', expname=''):

    logging.basicConfig(filename=log_fname, filemode='w', format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)

    logging.info('#-------------------------------------------------')
    logging.info('\n')
    logging.info('#TRAINING:'  + expname)
    logging.info('\n')
    logging.info('#-------------------------------------------------')
    logging.info('\n')

    mode = 'a'
    if not os.path.exists (log_fname): mode = 'w'
    logfd = open(log_fname, mode)

    comm = [binpath, '-l', str(lossf), '-c', str(cvalue)]

    if svmargs:
        comm.extend(svmargs.split(' '))

    comm.extend([trainset_fname, outmodel_fname])

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
def svm_test (binpath, model_fname, testset_fname, results_fname, log_fname, expname=''):

    mode = 'a'
    if not os.path.exists (log_fname): mode = 'w'

    logging.basicConfig(filename=log_fname, filemode=mode, format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)

    logging.info('#-------------------------------------------------')
    logging.info('\n')
    logging.info('#CLASSIFYING:'  + expname)
    logging.info('\n')
    logging.info('#-------------------------------------------------')
    logging.info('\n')

    logfd = open(log_fname, 'a')

    comm = [binpath, testset_fname, model_fname, results_fname]
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
## MAIN
#-------------------------------------------------------------------------------
def main (argv=None):

    if argv is None:
        argv = sys.argv

    parser = set_parser()

    try:
        args = parser.parse_args  ()
    except argparse.ArgumentError, exc:
        au.log.error (exc.message + '\n' + exc.argument)
        parser.error(str(msg))
        return -1

    outdir  = args.outdir.strip ()
    svmdir  = args.svmdir.strip ()
    expname = args.expname.strip()
    featsf  = args.feats.strip  ()
    testf   = args.testf.strip  ()
    cvalue  = args.cvalue
    nlparam = args.nlparam
    lossf   = args.lossf
    kernel  = args.kernel
    svmargs = args.svmargs.strip()
    redoing = args.redoing

    gridsearch = args.gridsearch

    if gridsearch:
        try:
            cgrid = np.float_(args.reggrid.strip().split(','))
            pgrid = np.float_(args.pargrid.strip().split(','))
        except Exception, err:
            au.log.error("Error parsing --reggrid or --pargrid argument.")
            au.log.error(str(err))
            return -1

    au.setup_logger(args.verbosity)

    #ROCArea optimization?
    rocarea_opt = False
    if lossf == 10:
        rocarea_opt = True

    #grid search parameters
    stratified = True
    ntimes     = 3

    #setting command paths
    aizko_svm = os.path.realpath(__file__)
    svm_class_bin = 'svm_perf_classify'
    svm_learn_bin = 'svm_perf_learn'
    if svmdir:
        svm_class_bin = svmdir + os.path.sep + svm_class
        svm_learn_bin = svmdir + os.path.sep + svm_learn

    train_fname = featsf
    test_fname  = testf

    #creating output filename
    if expname:
        ofname = os.path.basename(expname)
    else:
        ofname = au.remove_ext(os.path.basename(train_fname))

    if not expname:
        expname = os.path.basename(train_fname)

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

    #validation parameters
    bestc = cvalue

    #setting svmperf arguments
    if kernel == 1 or kernel == 2:
        bestp = nlparam

        if svmargs: svmargs += ' '
        svmargs += '-t ' + str(kernel)
        if   kernel == 1: svmargs += ' -d ' + str(bestp)
        elif kernel == 2: svmargs += ' -g ' + str(bestp)

    #gridsearch
    if gridsearch:
        if kernel == 0:
            bestc = au.get_best_c_param (aizko_svm, train_fname, cgrid, outdir, expname, ntimes, stratified, rocarea_opt, svmargs)

        elif kernel == 1 or kernel == 2:
            bestc, bestp = au.get_best_polyrbf_params (aizko_svm, train_fname, kernel, cgrid, pgrid, outdir, expname, ntimes, stratified, rocarea_opt, svmargs)

    #test
    svm_learn (svm_learn_bin, train_fname, modelf, logf, bestc, lossf, svmargs, expname)

    svm_test  (svm_class_bin, modelf, test_fname, predsf, logf, expname)

    try:
        results = au.read_svmperf_results (logf, predsf, testlabels)

        np.savetxt (resf, results, fmt='%f')

        print ('The results are in ' + logf + ' and also in the same order in ' + resf)
        return 0

    except IOError, err:
        au.log.error(err)
        return -1

#-------------------------------------------------------------------------------
## END MAIN
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
