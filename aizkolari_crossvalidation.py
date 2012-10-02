#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#classf='/home/alexandre/Dropbox/Documents/phd/work/fepeco/test/classes'
#subjsf='/home/alexandre/Dropbox/Documents/phd/work/fepeco/test/subslist'
#mask='/home/alexandre/Dropbox/Documents/phd/work/fepeco/test/data/mask.nii.gz'
#outdir='/home/alexandre/Desktop/out'
#balanced=1
#folds=10
#./aizkolari_crossvalidation.py -c $classf -s $subjsf -k $folds -o $outdir -b $balanced

#classf='/home/alexandre/Desktop/aizkotest/classes'
#subjsf='/home/alexandre/Desktop/aizkotest/subjslist'
#outdir='/home/alexandre/Desktop/aizkotest/out'
#balanced=1
#folds=10
#./aizkolari_crossvalidation.py -c $classf -s $subjsf -k $folds -o $outdir -b $balanced

import argparse, os
import numpy as np
#import shutil

from cvpartition import cvpartition

def file_len(fname):
    f = open(fname)
    return len(f.readlines())

#-------------------------------------------------------------------------------

def main(argv=None):

    parser = argparse.ArgumentParser(description='Creates text files with the same number of lines as the subjs file with 0s and 1s indicating which ones go to the training set (0) or test set(1)')
    parser.add_argument('-c','--classes', dest='classes', required=True, help='class label file. one line per class: <class_label>,<class_name>.')
    parser.add_argument('-s','--subjs',  dest='subjs', required=True, help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
    parser.add_argument('-k','--folds', dest='folds', type=int, default=10, required=False, help='Number of folds to separate the data. Set to 0 if you want a leave-one-out.')
    parser.add_argument('-o','--out', dest='outdir', required=True, help='name of the output directory where the results will be put.')
    parser.add_argument('-b','--balanced', dest='balanced', default='1', choices=['1','0'], required=False, help='If 1 it will separate proportional number of subjects for each class, else it will randomly pick any subject from the list (default: 1)')

    args     = parser.parse_args()

    classf   = args.classes.strip()
    subjsf   = args.subjs.strip()
    outdir   = args.outdir.strip()
    folds    = args.folds
    balanced = args.balanced.strip()

    #reading label file
    labels     = []
    classnames = []

    labfile = open(classf, 'r')
    for l in labfile:
        line = l.strip().split(',')
        labels    .append (int(line[0]))
        classnames.append (line[1])

    labfile.close()

    labels     = np.array (labels)
    classnames = np.array (classnames)

    #reading subjects list
    subjlabidx = []
    subjs      = []
    subjfile   = open(subjsf, 'r')
    for s in subjfile:
        line = s.strip().split(',')
        lab = int(line[0])
        idx = np.where(labels == lab)[0]
        subjlabidx.append(idx[0])
        subjs.append     (line[1])

    subjfile.close()

    #transforming from list to vector
    subjlabidx = np.array (subjlabidx)
    subjs      = np.array (subjs)

    classnum = labels.size
    subjsnum = subjlabidx.size

    #if output dir does not exist, create
    if not(os.path.exists(outdir)):
        os.mkdir(outdir)

#copying input files to outdir
#shutil.copy (subjsf, outdir + os.path.sep + os.path.basename(subjsf))
#shutil.copy (classf, outdir + os.path.sep + os.path.basename(classf))

#saving the input files to the output folder
#outf_subjs  = outdir + os.path.sep + 'subjects'
#outf_labels = outdir + os.path.sep + 'labels'
#np.savetxt(outf_subjs,  subjs,      fmt='%s')
#np.savetxt(outf_labels, subjlabels, fmt='%i')

    #generating partitions
    if balanced:
        #gsiz[i] has number of subjects of group with label in idx i
        #gsiz    = np.empty(classnum, dtype=int)
        #gcvpart will have the partition for group i
        #cvparts will be iteratively filled with partition information for each group
        cvparts = np.empty([subjsnum, folds], dtype=int)
        for i in range(classnum):
            gsiz    = sum(subjlabidx == i)
            gcvpart = cvpartition (gsiz, folds)
        for f in range(folds):
            cvparts[subjlabidx == i,f] = gcvpart[:,f]

    else:
        cvparts = cvpartition(subjsnum, folds)

    #generating files
    np.savetxt(outdir + '/all.txt', np.column_stack([labels[subjlabidx],subjs]), fmt='%s,%s')

    for i in range(folds):
        part   = cvparts[:,i]
        fname = outdir + '/fold_'  + str(i+1).zfill(4) + '.txt'

        f = open(fname, 'w')
        f.write ('#subjects file name: ' + subjsf)
        f.write ('\n')
        f.write ('#number of subjects: ' + str(len(part)))
        f.write ('\n')
        f.write ('#fold number: ' + str(i+1))
        f.write ('\n')
        f.write ('#training set size: ' + str(sum(part==0)))
        f.write ('\n')
        f.write ('#training set label: 0')
        f.write ('\n')
        f.write ('#test set size: ' + str(sum(part==1)))
        f.write ('\n')
        f.write ('#test set label: 1')
        f.write ('\n')

        np.savetxt(f, part, fmt='%i')

        f.close()

    return 0

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())





