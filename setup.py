"""
BeamFit - Robust laser and charged particle beam image analysis
Copyright (C) 2020 Christopher M. Pierce (contact@chris-pierce.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
        version='1.2',
        description='Robust laser and charged particle beam image analysis',
        author='Christopher M. Pierce',
        author_email='contact@chris-pierce.com',
        python_requires='>=3.1',
        packages=setuptools.find_packages(),
        install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
        ],
        setup_requires=['numpy'],
        package_data={'': ['tests/test_data.pickle']},
        include_package_data=True,
        license='GNU Affero General Public License v3 or later (AGPLv3+)',
        classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
          "Development Status :: 4 - Beta",
          "Operating System :: OS Independent",
        ],
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt}
    )


# Setup the package
setuptools.setup(**metadata)
