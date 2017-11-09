from distutils.core import setup
from Cython.Build import cythonize
import numpy

# This performs both Cythonizing (translating to C) and compiling in one go.
setup(name='angdist',
      ext_modules = cythonize('angdist.pyx'),
      include_dirs=[numpy.get_include()])
#NB: on some platforms, the directory of numpy headers needs to specified explicitly as above.