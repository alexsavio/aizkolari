#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#TESTED ONLY ON BINARY CLASSIFICATION

#DEPENDENCIES:
#scikit-learn
#sudo apt-get install python-argparse python-numpy python-numpy-ext python-matplotlib python-scipy python-nibabel

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import re
import sys
import argparse
import subprocess
import logging as log
import numpy as np
import nibabel as nib
import shelve
import collections

#for pearson correlation
import scipy.stats as stats

#data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#classification
from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

#feature selection
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import matthews_corrcoef

#cross-validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold

#scores
from sklearn.metrics import auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

#pipelining
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

import aizkolari_utils as au

#-------------------------------------------------------------------------------
class Result (collections.namedtuple('Result', ['metrics', 'cl', 'prefs_thr', 'subjsf', 'presels', 'prefs', 'fs1', 'fs2'])):
    pass

#-------------------------------------------------------------------------------
def filter_objlist (olist, fieldname, fieldval):
    res = []
    for o in olist:
        if getattr(o, fieldname) == fieldval:
            res.append(o)

    return res

#-------------------------------------------------------------------------------
def classification_metrics (targets, preds, probs=None):

#    if probs != None and len(probs) > 0:
#        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1], 1)
#        roc_auc = auc_score(fpr, tpr)
#    else:
#        fpr, tpr, thresholds = roc_curve(targets, preds, 1)
#        roc_auc = auc_score(targets, preds)

    auc = 0
    if len(targets) > 1:
        auc = auc_score(targets, preds)

    cm = confusion_matrix(targets, preds)

    #accuracy
    acc = accuracy_score(targets, preds)

    #recall? True Positive Rate or Sensitivity or Recall
    sens = recall_score(targets, preds)

    #precision
    prec = precision_score(targets, preds)

    #f1-score
    f1 = f1_score(targets, preds, np.unique(targets), 1)

    tnr  = 0.0
    spec = 0.0
    #True Negative Rate or Specificity (tn / (tn+fp))
    if len(cm) == 2:
        if (cm[0,0] + cm[0,1]) != 0:
            spec = float(cm[0,0])/(cm[0,0] + cm[0,1])

    return acc, sens, spec, prec, f1, auc

#-------------------------------------------------------------------------------
def get_cv_classification_metrics (cv_targets, cv_preds, cv_probs=None):
    '''
    returns a matrix of size [n_folds x 6]
    where 6 are: acc, sens, spec, prec, f1, roc_auc
    '''

    if (isinstance(cv_targets, dict)):
        metrics = np.zeros((len(cv_targets.keys()), 6))
        rango   = cv_targets.keys()

        c = 0
        for i in rango:
            try:
                targets = cv_targets[i]
                preds   = cv_preds  [i]

                probs = None
                if len(cv_probs) > 0:
                    probs   = cv_probs  [i]
            except:
                print( "Unexpected error: ", sys.exc_info()[0] )
                debug_here()

            acc, sens, spec, prec, f1, roc_auc = classification_metrics (targets, preds, probs)
            metrics[c, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
            c += 1

    else:

        metrics = np.zeros((cv_targets.shape[0], 6))
        rango   = np.arange(cv_targets.shape[0])

        for i in rango:
            targsi = cv_targets[i,:]
            predsi = cv_preds  [i,:]

            if cv_probs != None: probsi = cv_probs[i,:,:]
            else: probsi = None

            acc, sens, spec, prec, f1, roc_auc = classification_metrics (targsi, predsi, probsi)

            metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])

    return metrics

#-------------------------------------------------------------------------------
def parse_subjects_list (fname, datadir=''):
    labels = []
    subjs  = []

    if datadir:
        datadir += os.path.sep

    try:
        f = open(fname, 'r')
        for s in f:
            line = s.strip().split(',')
            labels.append(np.float(line[0]))
            subjf = line[1].strip()
            if not os.path.isabs(subjf):
                subjs.append (datadir + subjf)
            else:
                subjs.append (subjf)
        f.close()

    except:
        au.log.error( "Unexpected error: ", sys.exc_info()[0] )
        sys.exit(-1)

    return [labels, subjs]

#-------------------------------------------------------------------------------
def create_subjects_file (subjs_list, labels, output):
    lines = []
    for s in range(len(subjs_list)):
        subj = subjs_list[s]
        lab  = labels[s]
        line =  str(lab) + ',' + subj
        lines.append(line)

    lines = np.array(lines)
    np.savetxt(output, lines, fmt='%s')

#-------------------------------------------------------------------------------
def load_data (subjsf, datadir, maskf, labelsf=None):

    #loading mask
    msk     = nib.load(maskf).get_data()
    n_vox   = np.sum  (msk > 0)
    indices = np.where(msk > 0)

    #reading subjects list
    [scores, subjs] = parse_subjects_list (subjsf, datadir)
    scores = np.array(scores)

    imgsiz  = nib.load(subjs[0]).shape
    dtype   = nib.load(subjs[0]).get_data_dtype()
    n_subjs = len(subjs)

    #checking mask and first subject dimensions match
    if imgsiz != msk.shape:
        au.log.error ('Subject image and mask dimensions should coincide.')
        exit(1)

    #relabeling scores to integers, if needed
    if not np.all(scores.astype(np.int) == scores):
    #    unis = np.unique(scores)
    #    scs  = np.zeros (scores.shape, dtype=int)
    #    for k in np.arange(len(unis)):
    #        scs[scores == unis[k]] = k
    #    y = scs.copy()
        le = LabelEncoder()
        le.fit(scores)
        y = le.transform(scores)
    else:
        y = scores.copy()

    y = y.astype(int)

    #loading data
    au.log.info ('Loading data...')
    X = np.zeros((n_subjs, n_vox), dtype=dtype)
    for f in np.arange(n_subjs):
        imf = subjs[f]
        au.log.info('Reading ' + imf)

        img = nib.load(imf).get_data()
        X[f,:] = img[indices]

    return X, y, scores, imgsiz, msk, indices 


#-------------------------------------------------------------------------------
def pearson_correlation (X, y):

    #number of features
    n_feats = X.shape[1]

    #creating output volume file
    p = np.zeros(n_feats)

    #calculating pearson accross all subjects
    for i in range(X.shape[1]):
        p[i] = stats.pearsonr (X[:,i], y)[0]

    p[np.isnan(p)] = 0

    return p


#-------------------------------------------------------------------------------
def distance_computation (X, y, dist_function):
    '''
    Apply any given 1-D distance function to X and y.
    Have a look at:
    http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    '''

    #number of features
    n_feats = X.shape[1]

    #creating output volume file
    p = np.zeros(n_feats)

    #calculating pearson accross all subjects
    for i in range(X.shape[1]):
        p[i] = dist_function (X[:,i], y)[0]

    p[np.isnan(p)] = 0

    return p
    
#-------------------------------------------------------------------------------
def bhattacharyya_dist (X, y):
    '''
    Univariate Gaussian Bhattacharyya distance between the groups in X, labeled by y.
    '''
    classes = np.unique(y)
    n_class = len(classes)
    n_feats = X.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = X[y == i, :]
                xj = X[y == j, :]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                si = np.std  (xi, axis=0)
                sj = np.std  (xj, axis=0)

                d  = 0.25 * (np.square(mi - mj) / (vi + vj)) + 0.5  * (np.log((vi + vj) / (2*si*sj)))
                d[np.isnan(d)] = 0
                d[np.isinf(d)] = 0

                b = np.maximum(b, d)

    return b


#-------------------------------------------------------------------------------
def welch_ttest (X, y):

    classes = np.unique(y)
    n_class = len(classes)
    n_feats = X.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = X[y == i, :]
                xj = X[y == j, :]
                yi = y[y == i]
                yj = y[y == j]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                n_subjsi = len(yi)
                n_subjsj = len(yj)

                t = (mi - mj) / np.sqrt((np.square(vi) / n_subjsi) + (np.square(vj) / n_subjsj))
                t[np.isnan(t)] = 0
                t[np.isinf(t)] = 0

                b = np.maximum(b, t)

    return b

#-------------------------------------------------------------------------------
def append_to_keys (mydict, preffix):
    return {preffix + str(key) : (transform(value) if isinstance(value, dict) else value) for key, value in mydict.items()}

#-------------------------------------------------------------------------------
def apply_distance_threshold (distances, thr, method='robust'):
    if   method == 'robust':     return au.robust_range_threshold (distances, thr)
    elif method == 'rank':       return au.rank_threshold         (distances, thr)
    elif method == 'percentile': return au.percentile_threshold   (distances, thr)

#-------------------------------------------------------------------------------
def pre_featsel (X, y, method, thr=95, dist_function=None, thr_method='robust'):
    '''
    INPUT
    X             : data ([n_samps x n_feats] matrix)
    y             : class labels
    method        : distance measure: 'pearson', 'bhattacharyya', 'welcht', ''
                    if method == '', will use dist_function
    thr           : percentile distance threshold
    dist_function : 
    thr_method    : method for thresholding: 'none', 'robust', 'ranking'

    OUTPUT
    m          : distance measure (thresholded or not)
    '''

    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        au.log.info ('Calculating Pearson correlation')
        m = np.abs(pearson_correlation (X, y))

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        au.log.info ('Calculating Bhattacharyya distance')
        m = bhattacharyya_dist (X, y)

    #Welch's t-test
    elif method == 'welcht':
        au.log.info ("Calculating Welch's t-test")
        m = welch_ttest (X, y)

    elif method == '':
        au.log.info ("Calculating distance between data and class labels")
        #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        m = distance_computation(X, y, dist_function)

    #if all distance values are 0
    if not m.any():
        au.log.info("No differences between groups have been found. Are you sure you want to continue?")
        return m

    #threshold data
    if thr_method != 'none':
        if thr_method == 'robust':
            mt = au.robust_range_threshold (m, thr)
        elif thr_method == 'percentile':
            mt = au.percentile_threshold (m, thr)
        elif thr_method == 'rank':
            mt = au.rank_threshold (m, thr)

        return mt

    return m



#-------------------------------------------------------------------------------
def get_clfmethod (clfmethod, n_feats, n_subjs):

    #classifiers
    classifiers = { 'cart'   : tree.DecisionTreeClassifier(random_state = 0),
                    'rf'     : RandomForestClassifier(max_depth=None, min_samples_split=1, random_state=None),
                    'gmm'    : GMM(init_params='wc', n_iter=20, random_state=0),
                    'rbfsvm' : SVC (probability=True, max_iter=50000, class_weight='auto'),
                    'polysvm': SVC (probability=True, max_iter=50000, class_weight='auto'),
                    'linsvm' : LinearSVC (class_weight='auto'),
                    'sgd'    : SGDClassifier (fit_intercept=True, class_weight='auto', shuffle=True, n_iter = np.ceil(10**6 / 416), loss='modified_huber'),
                    'percep' : Perceptron (class_weight='auto'),
    }

    #Classifiers parameter values for grid search
    if n_feats < 10:
        max_feats = range(1, n_feats, 2)
    else:
        max_feats = range(1, 30, 4)
    max_feats.extend([None, 'auto', 'sqrt', 'log2'])

    clgrid =      { 'cart'   : dict(criterion = ['gini', 'entropy'], max_depth = [None, 10, 20, 30]),
                    'rf'     : dict(n_estimators = [3, 5, 10, 30, 50, 100], max_features = max_feats),
                    'gmm'    : dict(n_components = [2,3,4,5], covariance_type=['spherical', 'tied', 'diag'], thresh = [True, False] ),
                    #'svm'  : dict(kernel = ['rbf', 'linear', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0 = np.logspace(-3, 3, num=7, base=10)),
                    #'svm'    : dict(kernel = ['rbf', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0=np.logspace(-3, 3, num=7, base=10)),
                    'rbfsvm' : dict(kernel = ['rbf'],  C = np.logspace(-3, 3, num=7, base=10), gamma  = np.logspace(-3, 3, num=7, base=10)),
                    'polysvm': dict(kernel = ['poly'], C = np.logspace(-3, 3, num=7, base=10), degree = np.logspace(-3, 3, num=7, base=10)),
                    'linsvm' : dict(C = np.logspace(-3, 3, num=7, base=10)),
                    'sgd'    : dict(loss=['hinge', 'modified_huber', 'log'], penalty=["l1","l2","elasticnet"], alpha=np.logspace(-6, -1, num=6, base=10)),
                    'percep' : dict(penalty=[None, 'l2', 'l1', 'elasticnet'], alpha=np.logspace(-3, 3, num=7, base=10)),
    }

    return classifiers[clfmethod], clgrid[clfmethod]


#-------------------------------------------------------------------------------
def get_fsmethod (fsmethod, n_feats, n_subjs, n_jobs=1):

    #Feature selection procedures
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = { 'rfe'       : RFE(estimator=SVC(kernel="linear"), step=0.05, n_features_to_select=2),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                  'rfecv'     : RFECV(estimator=SVC(kernel="linear"), step=0.05, loss_func=auc_score), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)
                                #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                  'univariate': SelectPercentile(f_classif, percentile=5),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                  'fpr'       : SelectFpr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                  'fdr'       : SelectFdr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/feature_selection.html
                  'extratrees': ExtraTreesClassifier(n_estimators=50, max_features='auto', n_jobs=n_jobs, random_state=0), #compute_importances=True (default)

                  'pca'       : PCA(n_components='mle'),
                  'rpca'      : RandomizedPCA(random_state=0),
                  'lda'       : LDA(),
                                #http://scikit-learn.org/dev/auto_examples/feature_selection_pipeline.html
                  'anova'     : SelectKBest(f_regression, k=n_feats),
    }

    #feature selection parameter values for grid search
    max_feats = ['auto']
    if n_feats < 10:
        feats_to_sel = range(2, n_feats, 2)
        n_comps = range(1, n_feats, 2)
    else:
        feats_to_sel = range(2, 20, 4)
        n_comps = range(1, 30, 4)
    max_feats.extend(feats_to_sel)

    n_comps_pca = list(n_comps)
    n_comps_pca.extend(['mle'])

    fsgrid =    { 'rfe'       : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)], n_features_to_select = feats_to_sel),
                  'rfecv'     : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)]),
                  'univariate': dict(percentile = [1, 3, 5, 10]),
                  'fpr'       : dict(alpha = [1, 3, 5, 10]),
                  'fdr'       : dict(alpha = [1, 3, 5, 10]),
                  'extratrees': dict(n_estimators = [1, 3, 5, 10, 30, 50], max_features = max_feats),
                  'pca'       : dict(n_components = n_comps_pca, whiten = [True, False]),
                  'rpca'      : dict(n_components = n_comps, iterated_power = [3, 4, 5], whiten = [True, False]),
                  'lda'       : dict(n_components = n_comps),
                  'anova'     : dict(k = n_comps),
    }

    return fsmethods[fsmethod], fsgrid[fsmethod]

#-------------------------------------------------------------------------------
def get_cv_method (targets, cvmethod='10', stratified=True):
    '''
    Create cross-validation class

    Input:
    targets   : class labels set in the same order as in X
    cvmethod  : string of a number or number for a K-fold method, 'loo' for LeaveOneOut
    stratified: boolean indicating whether to use a Stratified K-fold approach

    Output:
    cv: Returns a class from sklearn.cross_validation
    '''
    #cross-validation

    n = len(targets)

    if stratified:
        if isinstance(cvmethod, int):
            return StratifiedKFold(targets, cvmethod)
        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return StratifiedKFold(targets, int(cvmethod))
    else:
        if isinstance(cvmethod, int):
            return KFold(n, cvmethod)

        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return KFold(n, int(cvmethod))

    if cvmethod == 'loo':
        return LeaveOneOut(n)

    return StratifiedKFold(targets, int(cvmethod))

#-------------------------------------------------------------------------------
def get_pipeline (fsmethod1, fsmethod2, clfmethod, n_subjs, n_feats, n_cpus):

    au.log.info('Preparing pipeline')

    combined_features = None
    if fsmethod1 != 'none' or fsmethod2 != 'none':
        #feature selection pipeline
        fs1n = fsmethod1
        fs2n = fsmethod2

        #informing user
        info = 'Selecting features: FSMETHOD1: ' + fs1n
        if fs2n != 'none':
            info +=', FSMETHOD2: ' + fs2n
        au.log.info(info)

        #union of feature selection processes
        fs1, fs1p = get_fsmethod (fs1n, n_feats, n_subjs, n_cpus)
        fs1p = append_to_keys(fs1p, fs1n + '__')
        if fs2n != 'none':
            fs2, fs2p = get_fsmethod (fs2n, n_feats, n_subjs, n_cpus)
            fs2p = append_to_keys(fs2p, fs2n + '__')

            combined_features = FeatureUnion([(fs1n, fs1), (fs2n, fs2)])
            fsp  = dict(fs1p.items() + fs2p.items())
        else:
            combined_features = FeatureUnion([(fs1n, fs1)])
            fsp = fs1p

    #classifier instance
    classif, clp = get_clfmethod (clfmethod, n_feats, n_subjs)
    #clp     = append_to_keys(clgrid[clfmethod], clfmethod + '__')

    #if clfmethod == 'gmm':
    #    classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
    #                     for i in xrange(n_class)])

    #creating pipeline
    if combined_features:
        pipe = Pipeline([ ('fs', combined_features), ('cl', classif) ])

        #arranging parameters for the whole pipeline
        clp = append_to_keys(clp, 'cl__')
        fsp = append_to_keys(fsp, 'fs__')
        params = dict(clp.items() + fsp.items())
    else:
        #pipe does not work 
        #pipe = Pipeline([ ('cl', classif) ])
        #arranging parameters for the whole pipeline
        #clp = append_to_keys(clp, 'cl__')
        pipe = classif
        params = clp


    return pipe, params

#-------------------------------------------------------------------------------
def extract_classify (X, y, scores, prefsmethod, prefsthr, fsmethod1, fsmethod2, 
                      clfmethod, cvmethod, stratified, stddize, 
                      thrmethod='robust', n_cpus=1):

    #classifiers
    #cgrid    = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
    #if nclass
    #perfmeas = ['Accuracy', 'Precision', 'Recall', 'F1', 'PRBEP', 'ROCArea', 'AvgPrec']

    #defining parameters for classifiers
    n_class    = len(np.unique(y))
    n_subjs    = X.shape[0]
    n_feats    = X.shape[1]
    n_selfeats = min(n_feats, int(np.floor(n_subjs*0.06)))

    cv = get_cv_method (y, cvmethod, stratified)

    presels    = {}
    preds      = {}
    probs      = {}
    truth      = {}
    best_pars  = {}
    importance = {}
    fc = 0
    for train, test in cv:
        au.log.info('Processing fold ' + str(fc))

        #data cv separation
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]

        #scaling
        #if clfmethod == 'linearsvc' or clfmethod == 'onevsonesvc':
        if stddize:
            au.log.info('Standardizing data')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform    (X_test)
            #[X_train, dmin, dmax] = au.rescale (X_train, scale_min, scale_max)
            #[X_test,  emin, emax] = au.rescale (X_test,  scale_min, scale_max, dmin, dmax)


        #PRE feature selection
        if prefsmethod != 'none':
            #sc_train = scores[train]
            presels[fc] = pre_featsel (X_train, y_train, prefsmethod, prefsthr, thrmethod)
            if not presels[fc].any():
                au.log.info('No feature survived the ' + prefsmethod + '(' + thrmethod + ': '+ str(prefsthr) + ')' + ' feature selection.')
                continue

            X_train = X_train[:, presels[fc] > 0]
            X_test  = X_test [:, presels[fc] > 0]

        pipe, params = get_pipeline (fsmethod1, fsmethod2, clfmethod, n_subjs, n_feats, n_cpus)

        #creating grid search
        gs = GridSearchCV (pipe, params, n_jobs=n_cpus, verbose=1)

        #do it
        au.log.info('Running grid search')
        gs.fit(X_train, y_train)

        au.log.info('Predicting on test set')

        #predictions, feature importances and best parameters
        preds     [fc] = gs.predict(X_test)
        truth     [fc] = y_test
        best_pars [fc] = gs.best_params_

        if hasattr(gs.best_estimator_, 'support_vectors_'):
            importance[fc] = gs.best_estimator_.support_vectors_
        elif hasattr(gs.best_estimator_, 'feature_importances_'):
            importance[fc] = gs.best_estimator_.feature_importances_

        if hasattr(gs.estimator, 'predict_proba'):
            try:
                probs [fc] = gs.predict_proba(X_test)
            except:
                probs [fc] = []

        #hello user
        au.log.info( 'Result: ' + str(y_test) + ' classified as ' + str(preds[fc]))

        fc += 1

    return preds, probs, best_pars, presels, cv, importance, scores, y, truth

