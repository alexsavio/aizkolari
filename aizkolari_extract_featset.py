#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-07-26
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, sys, argparse
import numpy as np
import nibabel as nib
import scipy.io as sio

import aizkolari_utils as au
import aizkolari_export as ae

def set_parser():
    parser = argparse.ArgumentParser(description='Saves a file with feature sets extracted from NIFTI files. The format of this file can be selected to be used in different software packages, including Numpy binary format, Weka, Octave/Matlab and SVMPerf.')
    parser.add_argument('-s', '--subjsf', dest='subjs', required=True, help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
    parser.add_argument('-o', '--outdir', dest='outdir', required=True,
                        help='''name of the output directory where the results will be saved. \n
                        In this directory the following files will be created:
                        - included_subjects: list of full path to the subjects included in the feature set.
                        - excluded_subjects: list of full path to the subjects excluded from the feature set. if any.
                        - included_subjlabels: list of class labels of each subject in included_subjects.
                        - excluded_subjlabels: list of class labels of each subject in excluded_subjects, if any.
                        - features.*: file containing a NxM matrix with the features extracted from subjects (N: subj number, M: feat number).
                      ''')
    parser.add_argument('-m', '--mask', dest='mask', required=True,
                        help='Mask file to extract feature voxels, any voxel with values > 0 will be included in the extraction.')
    parser.add_argument('-d', '--datadir', dest='datadir', required=False,
                        help='folder path where the subjects are, if the absolute path is not included in the subjects list file.', default='')
    parser.add_argument('-p', '--prefix', dest='prefix', default='', required=False,
                        help='Prefix for the output filenames.')
    parser.add_argument('-e', '--exclude', dest='exclude', default='', required=False,
                        help='subject list mask, i.e., text file where each line has 0 or 1 indicating with 1 which subject should be excluded in the measure. To help calculating measures for cross-validation folds.')
    parser.add_argument('-t', '--type', dest='type', default='numpybin', choices=['numpybin','octave','arff', 'svmperf'], required=False,
                        help='type of the output file. Alloweds: numpybin (Numpy binary file), octave (Octave/Matlab binary file using Scipy.io.savemat), arff (Weka text file), svmperfdat (.dat for SVMPerf).')
    parser.add_argument('-n', '--name', dest='dataname', default='aizkolari_extracted', required=False,
                        help='Name of the dataset. It is used for internal usage in SVMPerf and Weka.')
    parser.add_argument('-k', '--scale', dest='scale', default=False, action='store_true', required=False,
                        help='This option will enable Range scaling of the non-excluded data and save a .range file with the max and min of the scaled dataset to scale other dataset with the same transformation.')
    parser.add_argument('--scale_min', dest='scale_min', default=-1, type=int, required=False, help='Minimum value for the new scale range.')
    parser.add_argument('--scale_max', dest='scale_max', default= 1, type=int, required=False, help='Maximum value for the new scale range.')
    parser.add_argument('-r', '--thrP', dest='thresholdP', default='', required=False,
                        help='use following percentage (0-100) of ROBUST RANGE to threshold mask image (zero anything below the number). One or quoted list of floats separated by blank space.')
    parser.add_argument('-b', '--thr', dest='lthreshold', default='', required=False,
                        help='use following number to threshold mask image (zero anything below the number).')
    parser.add_argument('-u', '--uthr', dest='uthreshold', default='', required=False,
                        help='use following number to upper-threshold mask image (zero anything above the number).')
    parser.add_argument('-a', '--abs', dest='absolute', action='store_true', required=False,
                        help='use absolute value of mask before thresholding.')
    parser.add_argument('-l', '--leave', dest='leave', default=-1, required=False, type=int, help='index from subject list (counting from 0) indicating one subject to be left out of the training set. For leave-one-out measures.')
    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Progression reports, 2 for Debug reports')

    return parser

#-------------------------------------------------------------------------------
def get_out_extension (otype):
    if   otype  == 'numpybin':
        ext = au.numpyio_ext()
    elif otype  == 'octave':
        ext = au.octaveio_ext()
    elif otype  == 'svmperf':
        ext = au.svmperfio_ext()
    elif otype  == 'arff':
        ext = au.wekaio_ext()
    else:
        err = 'get_out_extension: Extension type not supported: ' + otype
        raise Exception(err)

    return ext

#-------------------------------------------------------------------------------
def get_filepath (outdir, filename, otype):

    filename = outdir + os.path.sep + filename
    try:
        filename += get_out_extension(otype)

    except Exception, err:
        au.log.error (str(err))
        sys.exit(-1)

    return filename

#-------------------------------------------------------------------------------
def rescale (data, range_min, range_max, data_min=np.NaN, data_max=np.NaN):
    if np.isnan(data_min):
        dmin = float(data.min())
    else:
        dmin = float(data_min)

    if np.isnan(data_max):
        dmax = float(data.max())
    else:
        dmax = float(data_max)

    try:
        factor = float(((range_max-range_min)/(dmax-dmin)) + ((range_min*dmax-range_max*dmin)/(dmax-dmin)))
        d = data*factor

    except Exception, err:
        au.log.error (str(err))
        sys.exit(-1)

    return d, dmin, dmax

#-------------------------------------------------------------------------------
def write_scalingrange_file (fname, dmin, dmax, scale_min, scale_max):
    f = open (fname, 'w')
    f.write('#data_min, data_max, range_min, range_max')
    f.write('\n')
    f.write(str(dmin) + ',' + str(dmax) + ',' + str(scale_min) + ',' + str(scale_max))
    f.close()

#-------------------------------------------------------------------------------
def save_data (outdir, prefix, dataname, otype, excluding, leave, feats, labels, exclfeats, exclulabels, dmin, dmax, scale, scale_min, scale_max, lthr, uthr, thrp, absolute):

    #setting output file name
    ofname = au.feats_str()

    if leave > -1:
        ofname += '.' + au.excluded_str() + str(leave)

    if absolute:  ofname += '.' + au.abs_str()
    if lthr:      ofname += '.lthr_' + str(lthr)
    if uthr:      ofname += '.uthr_' + str(uthr)
    if thrp:      ofname += '.thrP_' + str(thrp)
    if scale:     ofname += '.' + au.scaled_str()

    if excluding:
        excl_ofname  = au.excluded_str() + '_' + ofname
        exclfilename = get_filepath (outdir, excl_ofname , otype)

    if prefix:
        ofname = prefix + '_' + ofname
        excl_ofname = prefix + '_' + excl_ofname

    filename = get_filepath (outdir, ofname, otype)

    #writing in a text file the scaling values of this training set
    if scale:
        write_scalingrange_file (outdir + os.path.sep + ofname + '.scaling_range', dmin, dmax, scale_min, scale_max)

    #saving binary file depending on output type
    if otype == 'numpybin':
        np.save (filename, feats)

        if excluding:
            np.save (exclfilename, exclfeats)

    elif otype == 'octave':
        sio.savemat (filename, {au.feats_str(): feats, au.labels_str(): labels})
        if excluding:
            exclulabels[exclulabels == 0] = -1
            sio.savemat (exclfilename, {au.feats_str(): exclfeats, au.labels_str(): exclulabels})

    elif otype == 'svmperf':
        labels[labels == 0] = -1
        ae.write_svmperf_dat(filename, dataname, feats, labels)

        if excluding:
            exclulabels[exclulabels == 0] = -1
            ae.write_svmperf_dat(exclfilename, dataname, exclfeats, exclulabels)

    elif otype == 'arff':
        featnames = np.arange(nfeats) + 1
        ae.write_arff (filename, dataname, featnames, feats, labels)

        if excluding:
            ae.write_arff (exclfilename, dataname, featnames, exclfeats, exclulabels)

    else:
        err = 'Output method not recognised!'
        au.log.error(err)
        sys.exit(-1)

    return [filename, exclfilename]

#-------------------------------------------------------------------------------
def extract_features (subjs, exclusubjs, mask, maskf, scale, scale_min, scale_max):

    #population features
    nsubjs  = len(subjs)
    s       = nib.load(subjs[0])
    subjsiz = np.prod (s.shape)
    stype   = s.get_data_dtype()

    #loading subject data
    data = np.empty([nsubjs, subjsiz], dtype=stype)

    #number of voxels > 0 in mask
    mask   = mask.flatten()
    nfeats = np.sum(mask > 0)

    #reading each subject and saving the features in a vector
    feats = np.empty([nsubjs, nfeats], dtype=stype)

    #extracting features from non-excluded subjects
    c = 0
    for s in subjs:
        au.log.debug("Reading " + s)
        #check geometries
        au.check_has_same_geometry (s, maskf)
        #load subject
        subj = nib.load(s).get_data().flatten()
        #mask data and save it
        feats[c,:] = subj[mask > 0]
        c += 1

    #scaling if asked
    dmin = scale_min
    dmax = scale_max
    if scale:
        au.log.info("Scaling data.")
        [feats, dmin, dmax] = rescale(feats, scale_min, scale_max)

    #extracting features from excluded subjects
    exclfeats = []
    if exclusubjs:
        au.log.info("Processing excluded subjects.")
        nexcl     = len(exclusubjs)
        exclfeats = np.empty([nexcl, nfeats], dtype=stype)
        c = 0
        for s in exclusubjs:
            au.log.debug("Reading " + s)
            #check geometries
            au.check_has_same_geometry (s, maskf)
            #load subject
            subj = nib.load(s).get_data().flatten()
            #mask data and save it
            exclfeats[c,:] = subj[mask > 0]
            c += 1

        if scale:
            [exclfeats, emin, emax] = rescale(exclfeats, scale_min, scale_max, dmin, dmax)

    return [feats, exclfeats, dmin, dmax]

#-------------------------------------------------------------------------------
## START EXTRACT FEATSET
#-------------------------------------------------------------------------------
def main():

    #parsing arguments
    parser = set_parser()

    try:
        args = parser.parse_args ()
    except argparse.ArgumentError, exc:
        au.log.error (exc.message + '\n' + exc.argument)
        parser.error(str(msg))
        return -1

    subjsf     = args.subjs.strip   ()
    outdir     = args.outdir.strip  ()
    datadir    = args.datadir.strip ()
    excluf     = args.exclude.strip ()
    otype      = args.type.strip    ()
    dataname   = args.dataname.strip()

    maskf      = args.mask.strip()
    prefix     = args.prefix.strip()

    leave      = args.leave

    scale      = args.scale
    scale_min  = args.scale_min
    scale_max  = args.scale_max

    thrps      = args.thresholdP.strip().split()
    lthr       = args.lthreshold.strip()
    uthr       = args.uthreshold.strip()
    absolute   = args.absolute

    au.setup_logger(args.verbosity)

    #checking number of files processed
    if not os.path.exists(maskf): 
        err = 'Mask file not found: ' + maskf
        au.log.error(err)
        sys.exit(-1)

    #number of subjects
    subjsnum = au.file_len(subjsf)

    #reading subjects list
    subjlabels = np.zeros(subjsnum, dtype=int)
    subjslist  = {}
    subjfile   = open(subjsf, 'r')
    c = 0
    for s in subjfile:
        line = s.strip().split(',')
        subjlabels[c] = int(line[0])
        subjfname = line[1].strip()

        if not os.path.isabs(subjfname) and datadir:
            subjslist[c] = datadir + os.path.sep + subjfname
        else:
            subjslist[c] = subjfname

        c += 1

    subjfile.close()

    #excluding if excluf or leave > -1
    subjmask = []
    excluding = False
    if excluf:
        excluding = True
        subjmask  = np.loadtxt(excluf, dtype=int)
    else:
        subjmask  = np.zeros(subjsnum, dtype=int)
        if leave > -1:
            excluding = True
            subjmask[leave] = 1

    subjs       = [ subjslist[elem] for elem in subjslist  if subjmask[elem] == 0]
    labels      = subjlabels[subjmask == 0]

    exclusubjs  = [ subjslist[elem] for elem in subjslist  if subjmask[elem] == 1]
    exclulabels = subjlabels[subjmask == 1]
    if not excluding:
        exclusubjs = []

    #mask process
    au.log.info('Processing ' + maskf)

    #loading mask and masking it with globalmask
    mask   = nib.load(maskf).get_data()

    #thresholding
    if absolute:  mask = np.abs(mask)
    if lthr:      mask[mask < lthr] = 0
    if uthr:      mask[mask > uthr] = 0

    if thrps:
        for t in thrps:
            au.log.info ("Thresholding " + maskf + " with robust range below " + str(t) + " percent.")
            thrm = au.threshold_robust_range (mask, t)

            au.log.info ("Extracting features.")
            [feats, exclfeats, dmin, dmax] = extract_features (subjs, exclusubjs, thrm, maskf, scale, scale_min, scale_max)

            au.log.info ("Saving data files.")
            [filename, exclfilename] = save_data (outdir, prefix, dataname, otype, excluding, leave, feats, labels, exclfeats, exclulabels, dmin, dmax, scale, scale_min, scale_max, lthr, uthr, t, absolute)

    else:
        au.log.info ("Extracting features.")
        [feats, exclfeats, dmin, dmax] = extract_features (subjs, exclusubjs, mask, maskf, scale, scale_min, scale_max)

        au.log.info ("Saving data files.")
        [filename, exclfilename] = save_data (outdir, prefix, dataname, otype, excluding, leave, feats, labels, exclfeats, exclulabels, dmin, dmax, scale, scale_min, scale_max, lthr, uthr, thrps, absolute)

    au.log.info ("Saved " + filename)
    if excluding:
        au.log.info ("Saved " + exclfilename)

    #saving description files
    np.savetxt(filename + '.' + au.subjectfiles_str(), subjs,  fmt='%s')
    np.savetxt(filename + '.' + au.labels_str(),       labels, fmt='%i')

    if excluding:
        np.savetxt(exclfilename + '.' + au.subjectfiles_str(), exclusubjs,  fmt='%s')
        np.savetxt(exclfilename + '.' + au.labels_str(),       exclulabels, fmt='%i')

    return 1

#-------------------------------------------------------------------------------
## END EXTRACT FEATSET
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
