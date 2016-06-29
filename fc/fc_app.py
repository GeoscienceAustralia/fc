from __future__ import absolute_import, print_function

from datetime import datetime
import logging
from copy import deepcopy

import click
from pathlib import Path
from pandas import to_datetime

from datacube.api.grid_workflow import GridWorkflow
from datacube.ui import click as ui
from datacube.model import DatasetType, GeoPolygon, Range
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.utils import intersect_points, union_points, unsqueeze_dataset

from fc.fractional_cover import fractional_cover
from fc.task_app import task_app, task_app_options, get_full_lineage


_LOG = logging.getLogger('grid-app')


def make_fc_config(index, config, **query):
    dry_run = query.get('dry_run', False)

    source_type = index.products.get_by_name(config['source_type'])
    if not source_type:
        _LOG.error("Source DatasetType %s does not exist", config['source_type'])
        return 1

    output_type = DatasetType(source_type.metadata_type, deepcopy(source_type.definition))
    output_type.definition['name'] = config['output_type']
    output_type.definition['managed'] = True
    output_type.definition['description'] = config['description']
    output_type.definition['storage'] = config['storage']
    output_type.metadata['format'] = {'name': 'NetCDF'}

    var_param_keys = {'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'attrs'}

    output_type.definition['measurements'] = [{k: v for k, v in measurement.items() if k not in var_param_keys}
                                              for measurement in config['measurements']]

    chunking = config['storage']['chunking']
    chunking = [chunking[dim] for dim in config['storage']['dimension_order']]

    variable_params = {}
    for mapping in config['measurements']:
        varname = mapping['name']
        variable_params[varname] = {k: v for k, v in mapping.items() if k in var_param_keys}
        variable_params[varname]['chunksizes'] = chunking

    config['variable_params'] = variable_params

    if not dry_run:
        _LOG.info('Created DatasetType %s', output_type.name)
        output_type = index.products.add(output_type)

    config['nbar_dataset_type'] = source_type
    config['fc_dataset_type'] = output_type

    return config


def get_filename(config, tile_index, sources):
    file_path_template = str(Path(config['location'], config['file_path_template']))
    return file_path_template.format(tile_index=tile_index,
                                     start_time=to_datetime(sources.time.values[0]).strftime('%Y%m%d%H%M%S%f'),
                                     end_time=to_datetime(sources.time.values[-1]).strftime('%Y%m%d%H%M%S%f'))


def make_fc_tasks(index, config, **kwargs):
    input_type = config['nbar_dataset_type']
    output_type = config['fc_dataset_type']

    workflow = GridWorkflow(index, output_type.grid_spec)

    # TODO: Filter query to valid options

    query = {}
    if 'year' in kwargs:
        year = int(kwargs['year'])
        query['time'] = Range(datetime(year=year, month=1, day=1), datetime(year=year+1, month=1, day=1))

    tiles_in = workflow.list_tiles(product=input_type.name, **query)
    tiles_out = workflow.list_tiles(product=output_type.name, **query)

    # TODO: Move get_full_lineage & update_sources to GridWorkflow/Datacube/model?
    def update_sources(sources):
        return tuple(get_full_lineage(index, dataset.id) for dataset in sources)

    def update_tile(tile):
        for i in range(tile['sources'].size):
            tile['sources'].values[i] = update_sources(tile['sources'].values[i])
        return tile

    def make_task(tile, **kwargs):
        nbar = update_tile(tile.copy())
        task = {
            'nbar': nbar
        }
        task.update(kwargs)
        return task

    tasks = [make_task(tile, tile_index=key, filename=get_filename(config, tile_index=key, sources=tile['sources']))
             for key, tile in tiles_in.items() if key not in tiles_out]

    _LOG.info('%s tasks discovered', len(tasks))
    return tasks


def get_app_metadata(config):
    doc = {
        'lineage': {
            'algorithm': {
                'name': 'datacube-ingest',
                'version': config.get('version', 'unknown'),
                'repo_url': 'https://github.com/GeoscienceAustralia/fc.git',
                'parameters': {'configuration_file': config.get('app_config_file', 'unknown')}
            },
        }
    }
    return doc


def make_fc_tile(nbar):
    input_tile = nbar.squeeze('time').drop('time')
    data = fractional_cover(input_tile)
    output_tile = unsqueeze_dataset(data, 'time', nbar.time)
    return output_tile


def do_fc_task(config, task):
    measurements = ['green', 'red', 'nir', 'swir1', 'swir2']

    nbar = GridWorkflow.load(task['nbar'], measurements)

    fc_out = make_fc_tile(nbar)

    global_attributes = config['global_attributes']
    variable_params = config['variable_params']
    file_path = Path(task['filename'])
    output_type = config['fc_dataset_type']

    def _make_dataset(labels, sources):
        assert len(sources)
        geobox = task['nbar']['geobox']
        source_data = reduce(union_points, (dataset.extent.to_crs(geobox.crs).points for dataset in sources))
        valid_data = intersect_points(geobox.extent.points, source_data)
        dataset = make_dataset(dataset_type=output_type,
                               sources=sources,
                               extent=geobox.extent,
                               center_time=labels['time'],
                               uri=file_path.absolute().as_uri(),
                               app_info=get_app_metadata(config),
                               valid_data=GeoPolygon(valid_data, geobox.crs))
        return dataset
    sources = task['nbar']['sources']
    datasets = xr_apply(sources, _make_dataset, dtype='O')
    nbar['dataset'] = datasets_to_doc(datasets)

    write_dataset_to_netcdf(fc_out, global_attributes, variable_params, Path(file_path))

    return datasets

app_name = 'fc'


@click.command(name=app_name)
@ui.pass_index(app_name=app_name)
@click.option('--dry-run', is_flag=True, default=False, help='Check if everything is ok')
@click.option('--year', help='Limit the process to a particulr year')
@task_app_options
@task_app(make_config=make_fc_config, make_tasks=make_fc_tasks)
def fc_app(index, config, tasks, executor, dry_run, *ars, **kwargs):
    click.echo('Starting Fractional Cover processing...')

    results = []
    for task in tasks:
        click.echo('Running task: {}'.format(task))
        results.append(executor.submit(do_fc_task, config=config, task=task))

    successful = failed = 0
    for result in executor.as_completed(results):
        try:
            datasets = executor.result(result)
            for dataset in datasets.values:
                index.datasets.add(dataset)
            successful += 1
        except Exception as err:  # pylint: disable=broad-except
            _LOG.exception('Task failed: %s', err)
            failed += 1
            continue

    click.echo('%d successful, %d failed' % (successful, failed))

