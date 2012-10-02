Aizkolari
=========

Aizkolari is a set of tools in Python to measure distances from different groups of Nifti images (brain MRI in my case) to do a supervised feature selection for supervised classification without circularity.

It makes use of 
- Python: http://www.python.org/
- iPython: http://ipython.org/
- Numpy: http://numpy.scipy.org/
- Scipy: http://www.scipy.org/
- FSL: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- Nibabel: http://nipy.sourceforge.net/nibabel/
- Matplotlib: http://matplotlib.org/
- SVMPerf: http://www.cs.cornell.edu/people/tj/svm_light/old/svm_perf_v2.50.html


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
My experiment consisted of supervised feature extraction of huge amount of data, which didn't allow me to load everything on RAM memory. The main motivation to do this was being able to slice all volumes (masks as well), calculate the measures I wanted and then merge these slices again. I went a little bit further programming the extraction and exportation of the feature sets to svmperf, weka and octave and the cross-validation k-folding, grid search for classification and results summarizing.

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
