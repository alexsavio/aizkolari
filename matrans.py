#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#README:
#Transforms a NxN matrix volume (N^2 volumes in 4th dimension) into other measure maps. 
#You can make a list of measures and they will be applied in order. 
#A list of the implemented measures are listed below.

#Geodesic anisotropy equation was extracted from
#P. G. Batchelor et al. - A Rigorous Framework for Diffusion Tensor Calculus - Magnetic Resonance in Medicine 53:221-225 (2005)

# What is tensor denoising?
#Log-Euclidean tensor denoising was used to eliminate singular, negative definite, or rank-deficient tensors
#-------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import argparse, os, sys

from time import clock
import nibabel as nib
import numpy as np
from scipy.linalg import logm
from scipy.linalg.matfuncs import sqrtm
from numpy.linalg import det
from numpy.linalg import eigvals
from numpy.linalg import eigvalsh

#-------------------------------------------------------------------------------
#definining measure functions
def mylogm (v):
   return np.reshape(logm(v.reshape(N,N)), [1,N*N])
#-------------------------------------------------------------------------------

def mydet (v):
   return det(v.reshape(N,N))
#-------------------------------------------------------------------------------

def mytrace (v):
   return np.trace(v.reshape(N,N))
#-------------------------------------------------------------------------------

def myeigvals (v):
   return eigvals(v.reshape(N,N)).flatten()
#-------------------------------------------------------------------------------

def mymaxeigvals (v):
   return max (myeigvals(v))
#-------------------------------------------------------------------------------

def myeigvalsh (v):
   return eigvalsh(v.reshape(N,N)).flatten()
#-------------------------------------------------------------------------------

def mymaxeigvalsh (v):
   return max (myeigvalsh(v))
#-------------------------------------------------------------------------------

def mydeftensor (v):
   j = v.reshape([N,N])
   s = sqrtm(j.transpose()*j)
   return S.reshape([1,N*N])
#-------------------------------------------------------------------------------

def mygeodan (v):
   s = logm(v.reshape(N,N))
   return np.sqrt(np.trace(np.square(s - np.trace(s)/N * np.eye(N))))
#-------------------------------------------------------------------------------

def calculate_measures (funcs, data, odims):
   for i in range(len(funcs)):
      measure = funcs[i]
      odim    = odims[i]
      data    = measure(data)

   return data
#-------------------------------------------------------------------------------

def set_parser():
   parser = argparse.ArgumentParser(description='Transforms a NxN matrix volume (N^2 volumes in 4th dimension) into other measure maps. \n You can make a list of measures and they will be applied in order. \n A list of the implemented measures are listed below.', prefix_chars='-')
   parser.add_argument('-i', '--in', dest='infile', required=True,
                      help='Jacobian matrix volume (4DVolume with 9 volumes)')
   parser.add_argument('-m', '--mask', dest='maskfile', required=False,
                      help='Mask file')
   parser.add_argument('-o', '--out', dest='outfile', required=True,
                      help='Output file name')
   parser.add_argument('-N', '--dims', dest='dims', required=False, default=3, type=int,
                      help='Order of the matrices in the volume')
   parser.add_argument('--matlog', dest='funcs', action='append_const', const='matlog',
                        help='Matrix logarithm')
   parser.add_argument('--deftensor', dest='funcs', action='append_const', const='deftensor',
                        help='Deformation tensor S=sqrtm(J`*J)')
   parser.add_argument('--det', dest='funcs', action='append_const', const='det',
                        help='Determinant')
   parser.add_argument('--trace', dest='funcs', action='append_const', const='trace',
                        help='Trace')
   parser.add_argument('--eigvals', dest='funcs', action='append_const', const='eigvals',
                        help='Eigenvalues of a general matrix')
   parser.add_argument('--maxeigvals', dest='funcs', action='append_const', const='maxeigvals',
                        help='Maximum eigenvalue of a general matrix')
   parser.add_argument('--eigvalsh', dest='funcs', action='append_const', const='eigvalsh',
                        help='Eigenvalues of a Hermitian or real symmetric matrix')
   parser.add_argument('--maxeigvalsh', dest='funcs', action='append_const', const='maxeigvalsh',
                        help='Maximum eigenvalue of a Hermitian or real symmetric matrix')
   parser.add_argument('--geodan', dest='funcs', action='append_const', const='geodan',
                        help='Geodesic anisotropy: sqrt(trace(matlog(S) - (trace(matlog(S))/N)*eye(N))^2, where N==3 ')
   return parser


#Geodesic anisotropy from:
#COMPARISON OF FRACTIONAL AND GEODESIC ANISOTROPY IN DIFFUSION TENSOR IMAGES OF 90 MONOZYGOTIC AND DIZYGOTIC TWINS
#Agatha D. Lee1, Natasha Lepore1, Marina Barysheva1, Yi-Yu Chou1, Caroline Brun1, Sarah K. Madsen1, Katie L. McMahon2, 1 Greig I. de Zubicaray2, Matthew Meredith2, Margaret J. Wright3, Arthur W. Toga1, Paul M. Thompson
#http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.142.3274

#-------------------------------------------------------------------------------
## START MATRIX TRANSFORMATIONS
#-------------------------------------------------------------------------------
def main():

   #parsing arguments
   parser = set_parser()

   #parsing arguments
   try:
      args     = parser.parse_args  ()
   except argparse.ArgumentError, exc:
      print (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return -1

   ifile = args.infile.strip()
   ofile = args.outfile.strip()
   maskf = args.maskfile.strip()
   funcs = args.funcs

   #setting the global variable that indicates the order of the matrices
   global N
   N     = args.dims

   #loading file and preprocessing
   iinfo  = nib.load(ifile)
   affine = iinfo.get_affine()

   minfo = nib.load(maskf)

   if len(iinfo.shape) != 4:
      err = 'File ' + ifile + ' should be a 4D volume'
      print(err)
      return -1

   #global variable N (for the nested functions)
   N = np.sqrt(iinfo.shape[3])

   if not N % 1 == 0:
      err = 'File ' + ifile + ' should have N volumes along its 4th dimension, where N is an exponent of 2.'
      print(err)
      return -1

   try:
      #deciding what function to use
      # and indicating size of 4th dimension of output
      myfuncs = {}
      odims   = np.empty(len(funcs), dtype=int)

      for i in range(len(funcs)):
         if funcs  [i] == 'matlog':
            myfuncs[i] = mylogm
            odims  [i] = N
         elif funcs[i] == 'det':
            myfuncs[i] = mydet
            odims  [i] = 1
         elif funcs[i] == 'trace':
            myfuncs[i] = mytrace
            odims  [i] = 1
         elif funcs[i] == 'deftensor':
            myfuncs[i] = mydeftensor
            odims  [i] = N
         elif funcs[i] == 'eigvalsh':
            myfuncs[i] = myeigvalsh
            odims  [i] = 3
         elif funcs[i] == 'eigvals':
            myfuncs[i] = myeigvals
            odims  [i] = 3
         elif funcs[i] == 'maxeigvalsh':
            myfuncs[i] = myeigvalsh
            odims  [i] = 1
         elif funcs[i] == 'maxeigvals':
            myfuncs[i] = myeigvals
            odims  [i] = 1
         elif funcs[i] == 'geodan':
            myfuncs[i] = mygeodan
            odims  [i] = 1

      #reading input data
      img  = iinfo.get_data()
      mask = minfo.get_data()

      sx = img.shape[0]
      sy = img.shape[1]
      sz = img.shape[2]

      nvox = sx*sy*sz
      im   = img.reshape(nvox,9)
      msk  = mask.flatten()
      idx  = np.where(msk > 0)[0]

      tic = clock();

      #processing
      lm      = np.zeros([nvox, odims[-1]])
      for i in idx:
         lm[i,:] = calculate_measures (myfuncs, im[i,:], odims)
         #lm[i,:] = meafun(im[i,:])

      toc = clock() - tic
      print ('Time spent: ' + str(toc))

      #saving output
      lm = lm.reshape([sx, sy, sz, odims[-1]])
      lm = lm.squeeze()

   #   debug_here()
      new_image = nib.Nifti1Image(lm, affine)
      nib.save(new_image, ofile)

   except:
      print ('Ooops! Error processing file ' + ifile)
      print 'Unexpected error: ', sys.exc_info()
      return -1


if __name__ == "__main__":
    sys.exit(main())

#Testing multiprocessing. Not implemented. Leaving for patience to solve.
#for i in range(len(im)/7000):
#   p.apply_async(mylogm, args=(im[i,:],i))

##determining multiprocessing stuff
#if nthreads > 1:
#   from multiprocessing.pool import Pool

#ncpus = multiprocessing.cpu_count()
#if nthreads > ncpus:
#   nthreads = ncpus - 1

#if nthreads > 1:
#   p = ThreadPool(nthreads)
#   print ('Using ' + nthreads + ' threads for execution')

#import nibabel as nib
#import numpy as np
#from time import clock
#from multiprocessing import Pool
#ifile='patient.M.90..5.OAS1_0247_MR1_mpr_n4_anon_111_t88_masked_gfc_spline_jacmat.nii.gz'
#meta = nib.load(ifile)
#img = meta.get_data()
#sx = img.shape[0]
#sy = img.shape[1]
#sz = img.shape[2]
#im = img.reshape(sx*sy*sz,9)
#p = Pool(4)
#p.map(mylogm, im)

#from time import clock
#for l in range(3): 
#   ti = clock(); 
#   p = ThreadPool(4)
#   lm = im[np.arange(len(im)/500),:]
#   lm = np.zeros(lm.shape)
#   lm = p.map(mylogm, im)
##   for i in range(len(im)/500): 
##      v = p.apply_async(mylogm, args=(im[i,:]))
##   p.close()
##   p.join()
#   tf = clock()-ti
#   print tf

#for l in range(3):
#   lm = np.empty(im.shape)
#   ti = clock(); 
#   for i in range(len(im)/500): 
#      lm[i,:] = mylogm(im[i,:])
#   tf = clock()-ti
#   print tf
