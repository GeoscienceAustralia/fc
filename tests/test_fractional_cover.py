"""
Test functions for Fractional Cover App components
"""
import datetime
import os
from pathlib import Path

import xarray as xr

import datacube.utils.geometry
from datacube.model import Measurement
from fc.fc_app import tif_filenames, all_files_exist, _get_filename
from fc.fractional_cover import fractional_cover


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
    ds = xr.open_dataset(file_path, mask_and_scale=False, drop_variables='crs', **kwargs)
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
    assert yml == Path(base + '.yml').absolute()


def test_all_files_exist():
    current = os.path.realpath(__file__)
    filenames = [current, 'this_isnt_.here']
    assert not all_files_exist(filenames)
    filenames = [current, current]
    assert all_files_exist(filenames)

    filenames_dict = {'a': current, 'c': current, 'b': 'this_isnt_.here'}
    assert not all_files_exist(filenames_dict.values())
    filenames_dict = {'a': current, 'b': current}
    assert all_files_exist(filenames_dict.values())


def test_get_filename():
    class Fake(object):
        pass

    source = Fake()
    source.metadata = Fake()
    source.metadata.region_code = '097045'
    source.time = Fake()
    source.time.values = (datetime.date(2019, 4, 13), datetime.date(2019, 4, 14))
    template = 'LS8_OLI_FC/{region_code}_{start_time}_v{version}.nc'
    config = {'root_dir_in_new_location': 'LS8_OLI_NBART',
              'task_timestamp': 'the_timestamp',
              'location': Path('/can/this/be/made/up'),
              'file_path_template': template}  # root_dir_in_new_location: 'LS8_OLI_NBART'

    result = _get_filename(config, source)
    actual = '/can/this/be/made/up/LS8_OLI_FC/097045_20190413000000000000_vthe_timestamp.nc'
    assert result == actual


def test_get_filename3():
    class Fake(object):
        pass

    source = Fake()
    source.time = Fake()
    source.time.values = (datetime.date(2019, 4, 13), datetime.date(2019, 4, 14))
    source.metadata = None
    source.local_uri = 'file///w-43/LS8_OLI_NBART_3577_15_-43_20190412234447000000_v1556301492.nc'
    template = '_{tile_index[0]}_{tile_index[1]}_'
    tile_index_regex = '^(.+)/LS8_OLI_NBART_[0-9]*_(?P<tile_index0>-?[0-9]*)_(?P<tile_index1>-?[0-9]*)_[_v0-9]*.*$'
    config = {'source_directory': 'LS8_OLI_NBART',
              'task_timestamp': 'the_timestamp',
              'location': Path('/can/this/be/made/up'),
              'file_path_template': template,
              'tile_index_regex': tile_index_regex}
    result = _get_filename(config, source)
    actual = '/can/this/be/made/up/_15_-43_'
    assert result == actual


def test_get_filename4():
    class Fake(object):
        pass

    source = Fake()
    source.metadata = Fake()
    source.metadata.region_code = '097045'
    source.time = Fake()
    source.time.values = (datetime.date(2019, 4, 13), datetime.date(2019, 5, 14))
    template = 'LS8_OLI_FC/{region_code}_{start_time}_v{version}.nc'
    template = '{epoch_start:%Y-%m-%d}_{epoch_end:%m}.nc'
    config = {'root_dir_in_new_location': 'LS8_OLI_NBART',
              'task_timestamp': 'the_timestamp',
              'location': Path('/can/this/be/made/up'),
              'file_path_template': template}  # root_dir_in_new_location: 'LS8_OLI_NBART'

    result = _get_filename(config, source)
    actual = '/can/this/be/made/up/2019-04-13_05.nc'
    assert result == actual
