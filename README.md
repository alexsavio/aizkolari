Aizkolari
=========

Aizkolari is a Python module distributed under the 3-Clause BSD license.

Aizkolari is a set of tools in Python to measure distances from different groups of Nifti images (brain MRI in my case) to do a supervised feature selection for supervised classification without circularity.

It makes use of 
- Python 2.7: http://www.python.org/
- iPython 0.12.1: http://ipython.org/
- Numpy 1.6.2: http://numpy.scipy.org/
- Scipy 0.11: http://www.scipy.org/
- FSL 5.0: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- Nibabel 1.3.0: http://nipy.sourceforge.net/nibabel/
- Matplotlib 1.1.1: http://matplotlib.org/
- SVMPerf 2.5: http://www.cs.cornell.edu/people/tj/svm_light/old/svm_perf_v2.50.html


#For Ubuntu users:
- Dependencies:
 sudo apt-get install python-argparse python-numpy python-numpy-ext python-matplotlib python-scipy python-nibabel

- For development:
 sudo apt-get install ipython python-nifti python-nitime

- Using SVMPerf:
 In order to use aizkolari_svmperf.py you will need to download and install SVMPerf. Aizkolari scripts that use svmperf have argument flags which you'll have to indicate if you don't have SVMPerf binaries in the same folder along the aizkolari Python files.

- Installing FSL:
 To install FSL on Ubuntu or Debian I recommend: http://neuro.debian.net/

#Motivation
An experiment for my PhD consisted of supervised feature extraction of a good amount of data that I could not load entirelly on RAM memory. I decided to slice all volumes (masks as well), calculate the measures I wanted for each slice across all subjects (now loading them in RAM) and then merging these results for each slice to obtain a whole volume with the result. For those familiar with FSL tools, my inspiration was the bedpostx scripts. 
Later I went a little bit further programming the extraction and exportation of the feature sets to svmperf, weka and octave and k-fold or leave-one-out cross-test, grid search for SVM parameter tuning (linear and RBF) and summarizing the results.

The name of the toolset comes from the Basque language and the idea of slicing the data: http://en.wikipedia.org/wiki/Aizkolaritza.
In Spain there is an expression: "cabeza de serrín" which literally means "head full of sawdust".

#Support
Any doubt on how to use it, any change or idea you would like to have implemented, please contact me:
http://www.ehu.es/ccwintco/index.php/Usuario:Alexsavio

#Reference:
- Please, cite this paper if you will be using this toolset:

Alexandre Savio, Manuel Graña - ''Deformation based feature selection for computer aided diagnosis of Alzheimer’s disease'' - Expert Systems with Applications
(http://dx.doi.org/10.1016/j.eswa.2012.09.009)

The work done here has been possible thanks to Prof. Manuel Graña and a FPI fellowship from the Government of the Basque Country.

Thank you very much.
