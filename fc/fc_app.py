from __future__ import absolute_import, print_function, division

import errno
import logging
import os
from copy import deepcopy
from functools import partial
from time import time as time_now
from math import ceil
from pathlib import Path

import click
from pandas import to_datetime

from datacube.api.grid_workflow import GridWorkflow
from datacube.model import DatasetType, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.ui import click as ui
from datacube.ui import task_app
from datacube.utils import unsqueeze_dataset
from digitalearthau.qsub import QSubLauncher, with_qsub_runner, norm_qsub_params
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
        _LOG.warning('Current user appears not have write access output location: %s', config['location'])

    config['nbar_dataset_type'] = source_type
    config['fc_dataset_type'] = output_type

    if 'task_timestamp' not in config:
        config['task_timestamp'] = int(time_now())

    return config


def get_filename(config, tile_index, sources):
    file_path_template = str(Path(config['location'], config['file_path_template']))
    return file_path_template.format(tile_index=tile_index,
                                     start_time=to_datetime(sources.time.values[0]).strftime('%Y%m%d%H%M%S%f'),
                                     end_time=to_datetime(sources.time.values[-1]).strftime('%Y%m%d%H%M%S%f'),
                                     version=config['task_timestamp'])


def make_fc_tasks(index, config, time_range=None, **kwargs):
    input_type = config['nbar_dataset_type']
    output_type = config['fc_dataset_type']

    workflow = GridWorkflow(index, output_type.grid_spec)

    tiles_in = workflow.list_tiles(product=input_type.name, time=time_range)
    tiles_out = workflow.list_tiles(product=output_type.name, time=time_range)

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
    fc_out = make_fc_tile(nbar, output_measurements, config.get('sensor_regression_coefficients'))

    def _make_dataset(labels, sources):
        assert sources
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
        index.datasets.add(dataset, sources_policy='skip')
        _LOG.info('Dataset added')


APP_NAME = 'datacube-fc'

# TODO: This is probably all messed up, depends how we get installed
ROOT_DIR = Path(__file__).absolute().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
SCRIPT_DIR = ROOT_DIR / 'scripts'

# pylint: disable=invalid-name
tag_option = click.option('--tag', type=str,
                          default='notset',
                          help='Unique id for the job')


@click.group(help='Datacube Fractional Cover')
def cli():
    pass


@cli.command(name='list', help='List installed Fractional Cover config files')
def list_configs():
    for cfg in CONFIG_DIR.glob('*.yaml'):
        print(cfg.name)


def estimate_job_size(num_tasks):
    """ Translate num_tasks to number of nodes and walltime
    """
    max_nodes = 25
    cores_per_node = 16
    task_time_mins = 5

    # TODO: Tune this code:
    # "We have found for best throughput 25 nodes can produce about 11.5 tiles per minute per node,
    # with a CPU efficiency of about 96%."
    if num_tasks < max_nodes * cores_per_node:
        nodes = ceil(num_tasks / cores_per_node / 4)  # If fewer tasks than max cores, try to get 4 tasks to a core
    else:
        nodes = max_nodes

    tasks_per_cpu = ceil(num_tasks / (nodes * cores_per_node))
    wall_time_mins = '{mins}m'.format(mins=(task_time_mins * tasks_per_cpu))

    return nodes, wall_time_mins


@cli.command(help='Kick off two stage PBS job')
@click.option('--project', '-P', default='u46')
@click.option('--queue', '-q', default='normal',
              type=click.Choice(['normal', 'express']))
@click.option('--year', type=str, help='Limit the process to a particular year')
@ui.verbose_option
@tag_option
@task_app.app_config_option
@task_app.save_tasks_option
def submit(app_config,
           project,
           queue,
           output_tasks_file,
           year,
           tag):

    _LOG.info('Tag: %s', tag)

    qsub = QSubLauncher(norm_qsub_params(
        {'project': project,
         'queue': queue,
         'name': 'fc-generate-{}'.format(tag),
         'mem': '4G',
         'noask': True,
         'wd': True,
         'ncpus': 1,
         'walltime': '1h',
         'extra_qsub_args': ['-W', 'umask=027']}))

    app_config = Path(app_config).absolute()
    output_tasks_file = Path(output_tasks_file).absolute()

    ret_code, qsub_stdout = qsub('generate',
                                 '-v', '-v',
                                 '--save-tasks', str(output_tasks_file),
                                 '--project', project,
                                 '--queue', queue,
                                 '--app-config', str(app_config),
                                 '--year', year,
                                 '--tag', tag)

    _LOG.info('Launched qsub: %d -> %s', ret_code, qsub_stdout)


@cli.command(help='Generate Tasks into file and Queue PBS job to process them')
@click.option('--project', '-P', default='u46')
@click.option('--queue', '-q', default='normal',
              type=click.Choice(['normal', 'express']))
@click.option('--year', 'time_range', callback=task_app.validate_year, help='Limit the process to a particular year')
@click.option('--no-qsub', 'no_qsub', is_flag=True, default=False, help="Skip submitting qsub for next step")
@tag_option
@task_app.app_config_option
@task_app.save_tasks_option
@ui.config_option
@ui.verbose_option
@ui.log_queries_option
@ui.pass_index(app_name=APP_NAME)
def generate(index,
             app_config,
             project,
             queue,
             output_tasks_file,
             time_range,
             no_qsub,
             tag):
    output_tasks_file = Path(output_tasks_file).absolute()

    config, tasks = task_app.load_config(index, app_config, make_fc_config, make_fc_tasks, time_range=time_range)

    num_tasks_saved = task_app.save_tasks(config, tasks, output_tasks_file)
    _LOG.info('Tag: %s', tag)
    _LOG.info('Found %d tasks', num_tasks_saved)

    nodes, walltime = estimate_job_size(num_tasks_saved)

    _LOG.info('Will request %d nodes and %s time', nodes, walltime)

    if no_qsub:
        _LOG.info('Quitting early as requested')
        return 0

    qsub = QSubLauncher(norm_qsub_params(
        {'project': project,
         'queue': queue,
         'name': 'fc-run-{}'.format(tag),
         'mem': 'small',
         'wd': True,
         'noask': True,
         'nodes': nodes,
         'walltime': walltime,
         'extra_qsub_args': ['-W', 'umask=027']}))

    ret_code, qsub_stdout = qsub('run',
                                 '-v', '-v',
                                 '--load-tasks', str(output_tasks_file),
                                 '--celery', 'pbs-launch',
                                 '--tag', tag)

    _LOG.info('Launched qsub: %d -> %s', ret_code, qsub_stdout)
    return ret_code


@cli.command(help='Actually process generated task file')
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@with_qsub_runner()
@task_app.load_tasks_option
@tag_option
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def run(index,
        dry_run,
        tag,
        qsub, runner,
        input_tasks_file=None,
        *args, **kwargs):

    _LOG.info('Starting Fractional Cover processing...')
    _LOG.info('Tag: %s', tag)

    if input_tasks_file:
        config, tasks = task_app.load_tasks(input_tasks_file)

    if dry_run:
        task_app.check_existing_files((task['filename'] for task in tasks))
        return 0

    task_func = partial(do_fc_task, config)
    process_func = partial(process_result, index)

    runner(tasks, task_func, process_func)


if __name__ == "__main__":
    cli()
