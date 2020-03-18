"""
Setup
-----

This compiles all the Fortran extensions.
"""
import os

from numpy.distutils.core import Extension, setup

unmix_ext = Extension(
    name='fc.unmix.unmiximage',
    sources=[
        'fc/unmix/unmiximage.f90',
        'fc/unmix/constants_NSWC.f90',
        'fc/unmix/nnls.f90',
        'fc/unmix/unmiximage.pyf',
    ],
    extra_f90_compile_args=['-static']
)
unmix_ext.optional = True  # For platforms without FORTRAN, we will fall back to a SciPy implementation

config_files = ['config/' + name for name in os.listdir('config')]
setup(
    name='fc',
    description='Geoscience Australia - Fractional Cover for Digital Earth Australia',
    long_description=open('README.rst', 'r').read(),
    license='Apache License 2.0',
    url='https://github.com/GeoscienceAustralia/fc',
    maintainer='Geoscience Australia',
    maintainer_email='earth.observation@ga.gov.au',
    packages=['fc', 'fc.unmix'],
    data_files=[
        ('fc/config/', config_files)],
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'numpy',
        'numexpr',
        'datacube',
        'click>=6.0',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'datacube-fc = fc.fc_app:cli',
        ]
    },
    ext_modules=[unmix_ext],
)
