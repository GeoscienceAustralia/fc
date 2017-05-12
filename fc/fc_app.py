from __future__ import absolute_import, print_function

import errno
import logging
import os
from copy import deepcopy
from functools import partial
import time

import click
from pandas import to_datetime
from pathlib import Path

from datacube.api.grid_workflow import GridWorkflow
from datacube.model import DatasetType, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.ui import click as ui
from datacube.ui import task_app
from datacube.utils import unsqueeze_dataset
from fc.fractional_cover import fractional_cover


_LOG = logging.getLogger('agdc-fc')


def make_fc_config(index, config, dry_run=False, **query):
    source_type = index.products.get_by_name(config['source_type'])
    if not source_type:
        _LOG.error("Source DatasetType %s does not exist", config['source_type'])
        return 1

    output_type_definition = deepcopy(source_type.definition)
    output_type_definition['name'] = config['output_type']
    output_type_definition['managed'] = True
    output_type_definition['description'] = config['description']
    output_type_definition['metadata']['format'] = {'name': 'NetCDF'}
    output_type_definition['metadata']['product_type'] = config.get('product_type', 'fractional_cover')

    output_type_definition['storage'] = {k: v for (k, v) in config['storage'].items()
                                         if k in ('crs', 'tile_size', 'resolution', 'origin')}

    var_def_keys = {'name', 'dtype', 'nodata', 'units', 'aliases', 'spectral_definition', 'flags_definition'}

    output_type_definition['measurements'] = [{k: v for k, v in measurement.items() if k in var_def_keys}
                                              for measurement in config['measurements']]

    chunking = config['storage']['chunking']
    chunking = [chunking[dim] for dim in config['storage']['dimension_order']]

    var_param_keys = {'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'attrs'}
    variable_params = {}
    for mapping in config['measurements']:
        varname = mapping['name']
        variable_params[varname] = {k: v for k, v in mapping.items() if k in var_param_keys}
        variable_params[varname]['chunksizes'] = chunking

    config['variable_params'] = variable_params

    output_type = DatasetType(source_type.metadata_type, output_type_definition)

    if not dry_run:
        _LOG.info('Created DatasetType %s', output_type.name)
        output_type = index.products.add(output_type)

    if not os.access(config['location'], os.W_OK):
        _LOG.warn('Current user appears not have write access output location: %s', config['location'])

    config['nbar_dataset_type'] = source_type
    config['fc_dataset_type'] = output_type

    if 'task_timestamp' not in config:
        config['task_timestamp'] = int(time.time())

    return config


def get_filename(config, tile_index, sources):
    file_path_template = str(Path(config['location'], config['file_path_template']))
    return file_path_template.format(tile_index=tile_index,
                                     start_time=to_datetime(sources.time.values[0]).strftime('%Y%m%d%H%M%S%f'),
                                     end_time=to_datetime(sources.time.values[-1]).strftime('%Y%m%d%H%M%S%f'),
                                     version=config['task_timestamp'])


def make_fc_tasks(index, config, time=None, **kwargs):
    input_type = config['nbar_dataset_type']
    output_type = config['fc_dataset_type']

    workflow = GridWorkflow(index, output_type.grid_spec)

    tiles_in = workflow.list_tiles(product=input_type.name, time=time)
    tiles_out = workflow.list_tiles(product=output_type.name, time=time)

    def make_task(tile, **task_kwargs):
        task = dict(nbar=workflow.update_tile_lineage(tile))
        task.update(task_kwargs)
        return task

    tasks = (make_task(tile, tile_index=key, filename=get_filename(config, tile_index=key, sources=tile.sources))
             for key, tile in tiles_in.items() if key not in tiles_out)
    return tasks


def get_app_metadata(config):
    doc = {
        'lineage': {
            'algorithm': {
                'name': 'datacube-fc',
                'version': config.get('version', 'unknown'),
                'repo_url': 'https://github.com/GeoscienceAustralia/fc.git',
                'parameters': {'configuration_file': config.get('app_config_file', 'unknown')}
            },
        }
    }
    return doc


def make_fc_tile(nbar, measurements, regression_coefficients):
    input_tile = nbar.squeeze('time').drop('time')
    data = fractional_cover(input_tile, measurements, regression_coefficients)
    output_tile = unsqueeze_dataset(data, 'time', nbar.time.values[0])
    return output_tile


def do_fc_task(config, task):
    measurements = ['green', 'red', 'nir', 'swir1', 'swir2']

    global_attributes = config['global_attributes']
    variable_params = config['variable_params']
    file_path = Path(task['filename'])
    output_type = config['fc_dataset_type']

    if file_path.exists():
        raise OSError(errno.EEXIST, 'Output file already exists', str(file_path))

    nbar_tile = task['nbar']
    nbar = GridWorkflow.load(nbar_tile, measurements)

    output_measurements = config['fc_dataset_type'].measurements.values()
    fc_out = make_fc_tile(nbar, output_measurements, global_attributes['sensor_regression_coefficients'])

    def _make_dataset(labels, sources):
        assert len(sources)
        dataset = make_dataset(product=output_type,
                               sources=sources,
                               extent=nbar.geobox.extent,
                               center_time=labels['time'],
                               uri=file_path.absolute().as_uri(),
                               app_info=get_app_metadata(config),
                               valid_data=GeoPolygon.from_sources_extents(sources, nbar.geobox))
        return dataset

    datasets = xr_apply(nbar_tile.sources, _make_dataset, dtype='O')
    fc_out['dataset'] = datasets_to_doc(datasets)

    write_dataset_to_netcdf(
        dataset=fc_out,
        filename=Path(file_path),
        global_attributes=global_attributes,
        variable_params=variable_params,
    )
    return datasets


def process_result(index, result):
    for dataset in result.values:
        index.datasets.add(dataset, skip_sources=True)
        _LOG.info('Dataset added')


APP_NAME = 'datacube-fc'


@click.command(name=APP_NAME)
@ui.pass_index(app_name=APP_NAME)
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option('--year', 'time', callback=task_app.validate_year, help='Limit the process to a particular year')
@click.option('--queue-size', type=click.IntRange(1, 100000), default=3200,
              help='Number of tasks to queue at the start')
@task_app.task_app_options
@task_app.task_app(make_config=make_fc_config, make_tasks=make_fc_tasks)
def fc_app(index, config, tasks, executor, dry_run, queue_size, *args, **kwargs):
    click.echo('Starting Fractional Cover processing...')

    if dry_run:
        task_app.check_existing_files((task['filename'] for task in tasks))
        return 0

    task_func = partial(do_fc_task, config)
    process_func = partial(process_result, index)

    task_app.run_tasks(tasks, executor, task_func, process_func, queue_size)


if __name__ == "__main__":
    fc_app()
