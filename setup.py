import setuptools


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
    )


# Setup the package
setuptools.setup(**metadata)
