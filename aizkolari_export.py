#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-01-31
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import sys
import numpy as np
import nibabel as nib
import scipy.io as sio

import aizkolari_utils as au

def load_features (feats_file, labels_file):
   try:
      ext = au.get_extension(feats_file)

      if   ext == au.numpyio_ext ():
         feats = np.load(feats_file)

      elif ext == au.octaveio_ext():
         data   = sio.loadmat(feats_file)
         feats  = data['feats']
         labels = data['labels']

      elif ext == au.wekaio_ext  ():
         feats, meta = sio.arff.loadarff (featsfile)

      else:
         err = 'File extension not recognised: ' + feats_file
         raise IOError(err)

   except IOError:
      au.log.error("File not found: ", sys.exc_info()[0])

   except:
      au.log.error("Unexpected error:", sys.exc_info()[0])
      raise

   if labels_file:
      try:
         labels = np.loadtxt(labels_file)
      except IOError:
         au.log.error("File not found: ", labels_file)
      except:
         au.log.error("Unexpected error:", sys.exc_info()[0])
         raise

   return feats, labels

#-------------------------------------------------------------------------------
def write_svmperf_dat(filename, dataname, data, labels):
   """ ARFFWRITE  Writes numeric data as an SVM Perf .dat formatted file.

    USAGE:
          writeSVMPerfDAT(fileName,dataName,data,labels);
      example: writeSVMPerfDAT('myDB.arff','db-name',mydata,labels);

    INPUT:    
          filename:       String. Out file name.
          dataname:       String. A name for the database.
          data:           Numeric data matrix.
          labels:         Vector that indicates class, which must be {1,-1} and 
                          length as rows of data.

    DETAILS:
          Writes data using 4 digits to the right of the decimal point.

    EXAMPLE:

          // Having face images in vector-rows in a matrix
          [m n]=size(images);
          attnames=[1:n]; // If attribute names are trivial, just ennumerate them
          images('ImagesForSVMPerf.dat','Face Images',images,labels);

    SVM Perf .dat syntax:
   <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
   <target> .=. {+1,-1}
   <feature> .=. <integer>
   <value> .=. <float>
   <info> .=. <string> 
   """

   nsamps = data.shape[0]
   nfeats = data.shape[1]
   nlabs  = len(labels)
   if nlabs != nsamps:
      err = 'dimensions (rows) of data -1 must agree with number of labels!';
      raise IOError(err)

   label_chk = False
   if nlabs > 1:
       label_chk = sum((labels == 1) + (labels == -1));
       label_chk = label_chk != nsamps
   elif nlabs == 1:
       label_chk = labels[0] == 1 or labels[0] == -1

   if not label_chk:
      err = 'labels vector should have only -1 or 1 values!';
      raise IOError(err)

   # Open/create file
   fd = open(filename, 'w');

   # Write headings
   fd.write('#' + dataname + '\n');

   # Writing format for the data (comma delimited matrix)
   nfeats = data.shape[1]

   format = '%+d ';
   for i in np.arange(nfeats):
      format = format + str(i+1) + ':%6.4f ';

   labels = labels.reshape(nsamps, 1)
   data = np.concatenate((labels, data), axis=1)

   # Write data
   np.savetxt (fd, data, format);
   fd.close();

#-------------------------------------------------------------------------------
def write_arff(filename, dataname, featnames, data, labels):
   """
   # ARFFWRITE  Writes numeric data as an arff formatted file.

    USAGE:
          writeARFF(fileName,dataName,attNames,data);
      example: writeARFF('myDB.arff','db-name',myattnames,mydata);

    INPUT:
          filename:       String. Out File name.
          dataname:       String. A name for the database.
          featnames:      Array of numbers or cell of strings. Names of each
                          attribute.    
          data:           Numeric data matrix.
          labels:         Vector that indicates class


    DETAILS:
          Writes data using 4 digits to the right of the decimal point.

    EXAMPLE:

          // Having face images in vector-rows in a matrix
          [m n]=size(images);
          attnames=[1:n]; // If attribute names are trivial, just ennumerate them
          images('ImagesForWeka.arff','Face Images',attnames,images);

    ARFF (Attribute-Relation File Format) syntax:
    http://weka.wikispaces.com/ARFF+%28stable+version%29
   """

   # Check for input data
   nsamps = data.shape[0];
   nfeats = data.shape[1];
   if nfeats != len(featnames):
      err = 'dimensions (column) of data must agree with number of variable name!';
      raise IOError(err)

   # Open/create file
   fd = open(filename,'w');

   #Write headings
   fd.write('@RELATION ' + dataname + '\n');

   # Writing feature names in the arff file format.
   for i in featnames:
      fd.write('@ATTRIBUTE ' + str(i) + ' NUMERIC\n')

   # Write classes
   #classes = str(np.unique(labels)).replace('[','').replace(']','');
   classes = np.unique(labels).astype(int)
   classes = classes.reshape(1,len(classes))
   fd.write   ('@ATTRIBUTE class {')
   np.savetxt (fd, classes, fmt='%d', delimiter=',', newline='}')
   fd.write   ('\n')

   # Write data
   fd.write('@DATA\n');

   # Writing format for the data (comma delimited matrix)
   fmt = '';
   for i in featnames:
      fmt = fmt + ' %6.4f,';

   fmt = fmt + ' %d';

   labels = labels.reshape(nsamps, 1)
   data = np.concatenate((data, labels), axis=1)

   # Write data
   np.savetxt(fd, data, fmt);

   fd.close();

