"""
Setup
-----

This compiles all the Fortran extensions.
"""

from numpy.distutils.core import Extension, setup

unmix_ext = Extension(name='fc.unmix.unmiximage',
                    sources=[
                        'fc/unmix/unmiximage.f90',
                        'fc/unmix/constants_NSWC.f90',
                        'fc/unmix/nnls.f90',
                    ],
                    f2py_options=['only:', 'unmiximage',  ':'])

setup(name='fc',
      version='2.0',
      description='Geoscience Australia - Fractional Cover for AGDC',
      long_description=open('README.md', 'r').read(),
      license='Apache License 2.0',
      url='https://github.com/GeoscienceAustralia/fc',
      author='AGDC Collaboration',
      maintainer='AGDC Collaboration',
      maintainer_email='',
      packages=[
          'fc'
      ],
      install_requires=[
          'datacube',
      ],
      ext_modules=[
          unmix_ext,
      ],
      entry_points={
          'console_scripts': [
              'datacube-fc = fc.cli_app:cli',
          ]
      }
      )
