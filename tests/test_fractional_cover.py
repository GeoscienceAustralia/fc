# coding=utf-8
"""
Module
"""
from __future__ import absolute_import, print_function

import pytest
import os
from datetime import datetime
import xarray as xr
import datacube.utils.geometry
from datacube.model import Measurement
from fc.fractional_cover import fractional_cover
from fc.fc_app import tif_filenames, all_files_exist

from pathlib import Path


def test_fractional_cover(sr_filepath, fc_filepath):
    # print(sr_filepath)
    # print(fc_filepath)

    sr_dataset = open_dataset(sr_filepath)

    measurements = [
        Measurement(name='PV', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='NPV', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='BS', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='UE', dtype='int8', nodata=-1, units='1'),
    ]

    fc_dataset = fractional_cover(sr_dataset, measurements)

    assert set(fc_dataset.data_vars.keys()) == {m['name'] for m in measurements}

    validation_ds = open_dataset(fc_filepath)

    assert validation_ds == fc_dataset

    assert validation_ds.equals(fc_dataset)


def open_dataset(file_path):
    ds = xr.open_dataset(file_path, mask_and_scale=False, drop_variables='crs')
    ds.attrs['crs'] = datacube.utils.geometry.CRS('EPSG:32754')
    return ds


def test_filename2tif_names():
    bands = ['BS', 'PV']
    base = 'yeah'
    ext = '.tif'
    filename = base + ext
    abs_paths, rel_files, yml = tif_filenames(filename, bands)
    key = 'BS'
    assert abs_paths[key] == Path(base + '_' + key + ext).absolute().as_uri()
    assert rel_files[key] == str(base + '_' + key + ext)
    assert yml == Path(base + '.yml').absolute().as_uri()


def test_all_files_exist():
    current = os.path.basename(__file__)
    filenames = [current, 'this_isnt_.here']
    assert not all_files_exist(filenames)
    filenames = [current, current]
    assert all_files_exist(filenames)

    filenames_dict = {'a': current, 'c': current, 'b': 'this_isnt_.here'}
    assert not all_files_exist(filenames_dict.values())
    filenames_dict = {'a': current, 'b': current}
    assert all_files_exist(filenames_dict.values())
