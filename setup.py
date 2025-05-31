from setuptools import setup, Extension
import numpy

# Define the C extension
gaussufunc_extension = Extension(
    "gaussufunc", sources=["src/gaussian.c"], include_dirs=[numpy.get_include()]
)

setup(ext_modules=[gaussufunc_extension])
