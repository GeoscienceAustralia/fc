"""
Test functions for Fractional Cover App components
"""
import datetime
import os
from pathlib import Path

import datacube.utils.geometry
import numpy as np
import xarray as xr
from fc.fractional_cover import (DEFAULT_MEASUREMENTS, LANDSAT_8_COEFFICIENTS,
                                 _apply_coefficients_for_band, fractional_cover)
from fc.virtualproduct import FC_MEASUREMENTS, FractionalCover


def test_virtual_product_interface(sr_filepath, fc_filepath):
    fc_vp = FractionalCover()
    sr_dataset = open_dataset(sr_filepath)
    fake_time = np.array([np.datetime64("2010-01-01")])
    sr_dataset = sr_dataset.expand_dims({'time': fake_time})
    fc_dataset = fc_vp.compute(sr_dataset)

    assert set(fc_dataset.data_vars.keys()) == {m['name'] for m in FC_MEASUREMENTS}

    validation_ds = open_dataset(fc_filepath)
    validation_ds = validation_ds.rename({name: name.lower() for name in validation_ds.data_vars})
    fc_dataset = fc_dataset.squeeze(dim=['time'], drop=True)

    assert validation_ds == fc_dataset


def test_fc_with_regression(sr_filepath, fc_filepath):
    output_regression_coefficients = {'pv': [2.77, 0.9481], 'bs': [2.45, 0.9499], 'npv': [-0.73, 0.9578]}
    sr_dataset = open_dataset(sr_filepath)
    fc_dataset = fractional_cover(sr_dataset, DEFAULT_MEASUREMENTS,
                                  output_regression_coefficients=output_regression_coefficients)

    validation_ds = open_dataset(fc_filepath)
    for var in validation_ds.data_vars:
        if var.lower() != "ue":
            validation_ds[var].data = (validation_ds[var] * output_regression_coefficients[var.lower()][1]
                                       + output_regression_coefficients[var.lower()][0]).clip(min=0).data
    assert validation_ds == fc_dataset


def test_coefficients():
    data = np.zeros((10, 10))
    data_tweaked = _apply_coefficients_for_band(data, 'swir2', LANDSAT_8_COEFFICIENTS, clip_after_regression=True)
    assert np.all(np.greater_equal(data_tweaked, 0))


def test_fractional_cover(sr_filepath, fc_filepath):
    sr_dataset = open_dataset(sr_filepath)

    fc_dataset = fractional_cover(sr_dataset, DEFAULT_MEASUREMENTS)

    assert set(fc_dataset.data_vars.keys()) == {m['name'] for m in DEFAULT_MEASUREMENTS}

    validation_ds = open_dataset(fc_filepath)

    assert validation_ds == fc_dataset

    assert validation_ds.equals(fc_dataset)


def test_fractional_cover_lazy(sr_filepath, fc_filepath):
    sr_dataset = open_dataset(sr_filepath, chunks={'x': 50, 'y': 50})

    fc_dataset = fractional_cover(sr_dataset, DEFAULT_MEASUREMENTS)

    assert fc_dataset.PV.data.dask
    assert fc_dataset.NPV.data.dask
    assert fc_dataset.BS.data.dask
    assert fc_dataset.UE.data.dask

    fc_dataset.load()

    assert set(fc_dataset.data_vars.keys()) == {m['name'] for m in DEFAULT_MEASUREMENTS}

    validation_ds = open_dataset(fc_filepath)

    assert validation_ds == fc_dataset

    assert validation_ds.equals(fc_dataset)


def open_dataset(file_path, **kwargs):
    ds = xr.open_dataset(file_path, mask_and_scale=False, drop_variables='crs', **kwargs)
    ds.attrs['crs'] = datacube.utils.geometry.CRS('EPSG:32754')
    return ds
