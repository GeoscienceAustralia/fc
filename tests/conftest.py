'''
py.test configuration plugin

This module defines any fixtures or other extensions to py.test to be used throughout the
tests in this and sub packages.
'''

import pytest
from pathlib import Path


@pytest.fixture
def sr_filepath(request):
    return str(Path(__file__).parents[0] / 'data' / 'sr.nc')


@pytest.fixture
def fc_filepath(request):
    return str(Path(__file__).parents[0] / 'data' / 'fc.nc')
