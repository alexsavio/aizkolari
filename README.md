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
In order to use aizkolari_svmperf.py you will need to download and install SVMPerf. It has argument flags which you'll have to indicate if you don't have SVMPerf binaries in the same folder along the aizkolari Python files.

#Support
Any doubt on how to use it, any change or idea you would like to have implemented, please contact me.

#Reference:
- Please, cite this paper if you will be using this toolset.
Alexandre Savio, Manuel Graña - 'Deformation based feature selection for computer aided diagnosis of Alzheimer’s disease' - Expert Systems with Applications
(http://dx.doi.org/10.1016/j.eswa.2012.09.009)

Thank you very much.
