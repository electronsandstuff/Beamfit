#!/usr/bin/env python

'''
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
'''

import setuptools
import numpy as np

# Add in my c-extension
ext_modules = [setuptools.Extension('beamfit.gaussufunc', sources = ['src/gaussian.c'],
    include_dirs = [np.get_include()]) ]

setuptools.setup(
        name='beamfit',
        version='1.0',
        description='Robust laser and charged particle beam image analysis',
        author='Christopher M. Pierce',
        author_email='contact@chris-pierce.com',
        python_requires='>=3.1',
        packages=['beamfit'],
        install_requires = [
          'numpy',
          'matplotlib',
          'scipy'
        ],
        ext_modules = ext_modules,
        license = 'GNU Affero General Public License v3 or later (AGPLv3+)',
        classifiers = [
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
          "Development Status :: 4 - Beta",
          "Operating System :: OS Independent",
        ],
    )
