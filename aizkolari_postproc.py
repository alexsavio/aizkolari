#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import sys
import numpy as np
import nibabel as nib

import aizkolari_utils as au

def get_stats_fnames (groupnames, outdir=''):

    if np.ndim(groupnames) == 0:
        groupnames = [groupnames]

    if outdir:
        outdir += outdir + os.path.sep

    mnames  = [au.sums_str(), au.mean_str(), au.var_str(), au.std_str()]
    ngroups = len(groupnames)

    statfnames = np.zeros ([ngroups, len(mnames)], dtype=np.dtype('a2000'))

    for g in np.arange(ngroups):
        group = groupnames[g]
        for m in np.arange(len(mnames)):
            measure = mnames[m]
            statfnames[g,m] = outdir + group + '_' + measure + au.ext_str()

    return [statfnames, mnames]

#-------------------------------------------------------------------------------
def merge_stats_slices (datadir, group):
    slice_str = au.slice_str()

    groupfregex = group + 's_' + slice_str + '_????' + '.'

    #This is a 4D volume with all subjects, it can be a big file, so I'm not creating it
    #merge_slices (datadir, groupfregex, group + 's')
    au.imrm(datadir + os.path.sep + groupfregex)

    [statfnames, mnames] = get_stats_fnames (group, outdir='')

    statfnames = statfnames[0]

    out = []
    for i in np.arange(len(statfnames)):
        fname  = statfnames[i]
        m      = mnames[i]

        regex  = group + 's_' + slice_str + '_????' + '_' + m
        o = merge_slices (datadir, regex , fname, datadir, cleanup=False)
        au.imrm(datadir + os.path.sep + regex)
        out.append(o)

    return out

#-------------------------------------------------------------------------------
def merge_slices (datadir, fileregex, outfname, outdir='', cleanup=True):

    if not outdir:
        outdir = datadir

    au.log.info ('Merging the ' + fileregex + ' files in ' + outdir)

    fregex = datadir + os.path.sep + fileregex
    imglob = ''
    imglob = au.exec_comm(['imglob', fregex])
    imglob = imglob.strip()

    outdata = ''
    if imglob:
        if os.path.isabs (outfname): outdata = outfname
        else:                        outdata = outdir + os.path.sep + outfname

        os.system('fslmerge -z ' + outdata + ' ' + imglob)
        if cleanup:
            au.imrm(fregex)
    else:
        au.log.error ('aizkolari_postproc: Error: could not find ' + fregex + ' in ' + datadir)

    return outdata

#-------------------------------------------------------------------------------
def group_stats (datadir, groupname, groupsize, outdir=''):

    lst = os.listdir(datadir)
    n   = au.count_match(lst, groupname + 's_' + au.slice_regex() + au.ext_str())

    if not outdir:
        outdir = datadir

    au.log.info ('Calculating stats from group ' + groupname + ' in ' + outdir)

    for i in range(n):
        slino = au.zeropad(i)
        dataf = datadir + os.path.sep + groupname + 's_' + au.slice_str() + '_' + slino + au.ext_str()
        volstats (dataf, groupname, groupsize, outdir)


#-------------------------------------------------------------------------------
def volstats (invol, groupname, groupsize, outdir=''):

    slicesdir = os.path.dirname(invol)

    if not outdir:
        outdir = slicesdir

    base      = os.path.basename(au.remove_ext(invol))
    outmeanf  = outdir + os.path.sep + base + '_' + au.mean_str()
    outvarf   = outdir + os.path.sep + base + '_' + au.var_str()
    outstdf   = outdir + os.path.sep + base + '_' + au.std_str()
    outsumsf  = outdir + os.path.sep + base + '_' + au.sums_str()

    vol = nib.load(invol).get_data()
    aff = nib.load(invol).get_affine()

    if not os.path.exists(outmeanf):
        mean = np.mean(vol, axis=3)
        au.save_nibabel(outmeanf, mean, aff)

    if not os.path.exists(outstdf):
        std = np.std(vol, axis=3)
        au.save_nibabel(outstdf, std, aff)

    if not os.path.exists(outvarf):
        var = np.var(vol, axis=3)
        au.save_nibabel(outvarf, var, aff)

    if not os.path.exists(outsumsf):
        sums = np.sum(vol, axis=3)
        au.save_nibabel(outsumsf, sums, aff)

    return [outsumsf,outmeanf,outvarf,outstdf]


#-------------------------------------------------------------------------------
def remove_subject_from_stats (meanfname, varfname, samplesize, subjvolfname, newmeanfname, newvarfname, newstdfname=''):

    meanfname    = au.add_extension_if_needed(meanfname,    au.ext_str())
    varfname     = au.add_extension_if_needed(varfname,     au.ext_str())
    subjvolfname = au.add_extension_if_needed(subjvolfname, au.ext_str())

    newmeanfname = au.add_extension_if_needed(newmeanfname, au.ext_str())
    newvarfname  = au.add_extension_if_needed(newvarfname,  au.ext_str())

    if newstdfname:
        newstdfname = au.add_extension_if_needed(newstdfname, au.ext_str())

    #load data
    n = samplesize

    meanv = nib.load(meanfname).get_data()
    varv  = nib.load( varfname).get_data()
    subjv = nib.load(subjvolfname).get_data()
    aff   = nib.load(meanfname).get_affine()

    #calculate new mean: ((oldmean*N) - x)/(N-1)
    newmean = meanv.copy()
    newmean = ((newmean * n) - subjv)/(n-1)
    newmean = np.nan_to_num(newmean)

    #calculate new variance: 
    # oldvar = (n/(n-1)) * (sumsquare/n - oldmu^2)
    # s = ((oldvar * (n/(n-1)) ) + oldmu^2) * n
    # newvar = ((n-1)/(n-2)) * (((s - x^2)/(n-1)) - newmu^2)
    s = varv.copy()
    s = ((s * (n/(n-1)) ) + np.square(meanv)) * n
    newvar = ((n-1)/(n-2)) * (((s - np.square(subjv))/(n-1)) - np.square(newmean))
    newvar = np.nan_to_num(newvar)

    #save nifti files
    au.save_nibabel (newmeanfname, newmean, aff)
    au.save_nibabel (newvarfname , newvar,  aff)

    #calculate new standard deviation: sqrt(newvar)
    if newstdfname:
        newstd = np.sqrt(newvar)
        newstd = np.nan_to_num(newstd)
        au.save_nibabel (newstdfname, newstd, aff)

#(distance_func, mdir, classnames, gsize, chkf, foldno, expname, absval, leave, exsubf, exclas)
#-------------------------------------------------------------------------------
def group_distance (measure_function, datadir, groups, groupsizes, chkf, absolute=False, outdir='', foldno='', expname='', exclude_idx=-1, exclude_subj='', exclude_subjclass=''):

    olddir = os.getcwd()

    if not outdir:
        outdir = datadir

    ngroups = len(groups)
    #matrix of strings of 2000 characters maximum, to save filepaths
    gfnames = np.zeros ([ngroups,3], dtype=np.dtype('a2000'))

    subject_excluded = False

    for g1 in range(ngroups):
        g1name = groups[g1]
        #mean1fname
        gfnames[g1,0] = datadir + os.path.sep + g1name + '_' + au.mean_str()
        #var1fname  
        gfnames[g1,1] = datadir + os.path.sep + g1name + '_' + au.var_str()
        #std1fname
        gfnames[g1,2] = datadir + os.path.sep + g1name + '_' + au.std_str()

        for g2 in range(g1+1, ngroups):
            g2name = groups[g2]
            gfnames[g2,0] = datadir + os.path.sep + g2name + '_' + au.mean_str()
            gfnames[g2,1] = datadir + os.path.sep + g2name + '_' + au.var_str()
            gfnames[g2,2] = datadir + os.path.sep + g2name + '_' + au.std_str()

            experiment = g1name + '_vs_' + g2name

            #check if exclude_subjclass is any of both current groups
            eg = -1
            if exclude_idx > -1:
                if   exclude_subjclass == g1name: eg = g2
                elif exclude_subjclass == g2name: eg = g1

            step = au.measure_str() + ' ' + measure_function.func_name + ' ' + experiment + ' ' + datadir

            #remove subject from stats
            if eg > -1:
                exclude_str = '_' + au.excluded_str() + str(exclude_idx)
                step       += exclude_str
                experiment += exclude_str

                if not au.is_done(chkf, step):
                    if not subject_excluded:
                        newmeanfname = gfnames[eg,0] + exclude_str
                        newvarfname  = gfnames[eg,1] + exclude_str
                        newstdfname  = gfnames[eg,2] + exclude_str

                        rstep = au.remove_str() + ' ' + au.subject_str() + ' ' + str(exclude_subj) + ' ' + au.fromstats_str() + ' ' + datadir
                        if not au.is_done(chkf, rstep):
                           #(meanfname, varfname, samplesize, subjvolfname, newmeanfname, newvarfname, newstdfname='')
                           remove_subject_from_stats (gfnames[eg,0], gfnames[eg,1], groupsizes[eg][1], exclude_subj, newmeanfname, newvarfname, newstdfname)
                           au.checklist_add (chkf, rstep)

                        gfnames[eg,0] += exclude_str
                        gfnames[eg,1] += exclude_str
                        gfnames[eg,2] += exclude_str

                        groupsizes[eg][1] -= 1

                        subject_excluded = True

            #calculating distance
            if not au.is_done(chkf, step):
                mean1fname = au.add_extension_if_needed (gfnames[g1,0], au.ext_str())
                mean2fname = au.add_extension_if_needed (gfnames[g2,0], au.ext_str())
                var1fname  = au.add_extension_if_needed (gfnames[g1,1], au.ext_str())
                var2fname  = au.add_extension_if_needed (gfnames[g2,1], au.ext_str())
                std1fname  = au.add_extension_if_needed (gfnames[g1,2], au.ext_str())
                std2fname  = au.add_extension_if_needed (gfnames[g2,2], au.ext_str())

                outfname = measure_function (mean1fname, mean2fname, var1fname, var2fname, std1fname, std2fname, groupsizes[g1][1], groupsizes[g2][1], experiment, outdir, exclude_idx)

                if absolute:
                    change_to_absolute_values (outfname)

                au.checklist_add (chkf, step)

                return outfname

#-------------------------------------------------------------------------------
def change_to_absolute_values (niifname, outfname=''):

    niifname = au.add_extension_if_needed(niifname, au.ext_str())

    if not outfname:
        outfname = niifname

    try:
        #load data
        vol = nib.load(niifname).get_data()
        aff = nib.load(niifname).get_affine()

        vol = np.abs(vol)

        #save nifti file
        au.save_nibabel (outfname, vol, aff)

    except:
        au.log.error ("Change_to_absolute_values:: Unexpected error: ", sys.exc_info()[0])
        raise


