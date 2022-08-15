import setuptools
from setuptools.command.build_ext import build_ext as _build_ext


# Hack to bootstrap numpy
class BuildExt(_build_ext):
    """to install numpy"""
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# Add in my c-extension
ext_modules = [setuptools.Extension('gaussufunc', sources=['src/gaussian.c'],)]


# Write out the pacakge metadata
metadata = dict(
        name='beamfit',
        version='1.13',
        description='Robust laser and charged particle beam image analysis',
        author='Christopher M. Pierce',
        author_email='contact@chris-pierce.com',
        python_requires='>=3.1',
        packages=setuptools.find_packages(),
        install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'tensorflow',
        ],
        setup_requires=['numpy'],
        package_data={'': ['tests/test_data.pickle']},
        include_package_data=True,
        license="BSD-3-Clause",
        classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
        ],
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt}
    )


# Setup the package
setuptools.setup(**metadata)
