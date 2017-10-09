"""
Setup
-----

This compiles all the Fortran extensions.
"""
import os

from numpy.distutils.core import Extension, setup, Command

import versioneer


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys, subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


my_cmdclass = versioneer.get_cmdclass()
my_cmdclass['test'] = PyTest

setup(
    name='fc',
    version=versioneer.get_version(),
    cmdclass=my_cmdclass,
    description='Geoscience Australia - Fractional Cover for Digital Earth Australia',
    long_description=open('README.rst', 'r').read(),
    license='Apache License 2.0',
    url='https://github.com/GeoscienceAustralia/fc',
    author='Geoscience Australia',
    maintainer='Geoscience Australia',
    maintainer_email='earth.observation@ga.gov.au',
    packages=['fc', 'fc.unmix'],
    install_requires=[
        'numpy',
        'numexpr',
        'datacube',
        'click>=5.0',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'datacube-fc = fc.fc_app:cli',
        ]
    },
    ext_modules=[
        Extension(
            name='fc.unmix.unmiximage',
            sources=[
                'fc/unmix/unmiximage.f90',
                'fc/unmix/constants_NSWC.f90',
                'fc/unmix/nnls.f90',
                'fc/unmix/unmiximage.pyf',
            ],
        )
    ],
)
