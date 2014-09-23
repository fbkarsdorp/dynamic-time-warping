from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

ext_modules = [Extension("_dtw",["_dtw.pyx"], include_dirs=np_inc)]

setup(name='dtw',
      version='0.1',
      description='Dynamic Time Warping',
      author='Folgert Karsdorp',
      author_email='fbkarsdorp@gmail.com',
      package_dir={'dtw':'.'},
      packages=['dtw'],
      cmdclass = {'build_ext' : build_ext},
      ext_modules = ext_modules
      )