# coding=utf-8
"""
Module
"""
from __future__ import absolute_import, print_function

import pytest
import xarray as xr
import datacube.utils.geometry
from datacube.model import Measurement
from fc.fractional_cover import fractional_cover


def test_fractional_cover(sr_filepath, fc_filepath):
    print(sr_filepath)
    print(fc_filepath)

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


def test_fractional_cover_lazy(sr_filepath, fc_filepath):
    print(sr_filepath)
    print(fc_filepath)

    sr_dataset = open_dataset(sr_filepath, chunks={'x': 50, 'y': 50})

    measurements = [
        Measurement(name='PV', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='NPV', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='BS', dtype='int8', nodata=-1, units='percent'),
        Measurement(name='UE', dtype='int8', nodata=-1, units='1'),
    ]

    fc_dataset = fractional_cover(sr_dataset, measurements)

    assert fc_dataset.PV.data.dask
    assert fc_dataset.NPV.data.dask
    assert fc_dataset.BS.data.dask
    assert fc_dataset.UE.data.dask

    fc_dataset.load()

    assert set(fc_dataset.data_vars.keys()) == {m['name'] for m in measurements}

    validation_ds = open_dataset(fc_filepath)

    assert validation_ds == fc_dataset

    assert validation_ds.equals(fc_dataset)


def open_dataset(file_path, **kwargs):
    ds = xr.open_dataset(file_path, mask_and_scale=False, drop_variables='crs',  **kwargs)
    ds.attrs['crs'] = datacube.utils.geometry.CRS('EPSG:32754')
    return ds
