"""
Setup
-----

This compiles all the Fortran extensions.
"""
import os

from numpy.distutils.core import Extension, setup

setup(name='fc',
      version=os.environ.get('version', 0.0),
      description='Geoscience Australia - Fractional Cover for AGDC',
      long_description=open('README.md', 'r').read(),
      license='Apache License 2.0',
      url='https://github.com/GeoscienceAustralia/fc',
      author='AGDC Collaboration',
      maintainer='AGDC Collaboration',
      maintainer_email='',
      packages=['fc', 'fc.unmix'],
      install_requires=[
          'numpy',
          'datacube',
          'click'
      ],
      entry_points={
          'console_scripts': [
              'datacube-fc = fc.fc_app:fc_app',
          ]
      },
      ext_modules=[
          Extension(name='fc.unmix.unmiximage',
                    sources=[
                        'fc/unmix/unmiximage.f90',
                        'fc/unmix/constants_NSWC.f90',
                        'fc/unmix/nnls.f90',
                        'fc/unmix/unmiximage.pyf',
                    ],
                    )]
      )

