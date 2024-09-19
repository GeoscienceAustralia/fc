"""
Setup
-----
"""

import os

# required to compile fortran code
from skbuild import setup

config_files = ["config/" + name for name in os.listdir("config")]
setup(
    name="fractional-cover",
    description="Geoscience Australia - Fractional Cover for Digital Earth Australia",
    long_description_content_type="text/x-rst",
    long_description=open("README.rst", "r").read(),
    license="Apache License 2.0",
    url="https://github.com/GeoscienceAustralia/fc",
    maintainer="Geoscience Australia",
    maintainer_email="earth.observation@ga.gov.au",
    # We need the following two lines to be able to call `python setup.py sdist` to build a versioned sdist
    packages=["fc", "fc.unmix"],
    data_files=[("fc/config/", config_files)],
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["numpy", "numexpr", "datacube", "click>=6.0", "pandas"],
    entry_points={
        "console_scripts": [
            "datacube-fc = fc.fc_app:cli",
        ]
    },
)
