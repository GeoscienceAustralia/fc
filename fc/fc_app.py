from __future__ import absolute_import, print_function, division

import errno
import logging
import os
from copy import deepcopy
from datetime import datetime
from functools import partial
from time import time as time_now
from math import ceil
from pathlib import Path

import click
import sys

from pandas import to_datetime
from typing import Tuple

from dateutil import tz

from datacube.api.grid_workflow import GridWorkflow
from datacube.api.query import Query
from datacube.model import DatasetType, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.ui import click as ui
from datacube.ui import task_app
from datacube.utils import unsqueeze_dataset
from digitalearthau import paths, serialise
from digitalearthau.qsub import QSubLauncher, with_qsub_runner, norm_qsub_params, TaskRunner
from digitalearthau.runners.model import TaskAppState, TaskDescription, DefaultJobParameters
from fc.fractional_cover import fractional_cover
from datacube.index._api import Index

_LOG = logging.getLogger('agdc-fc')


def make_fc_config(index, config, dry_run=False, **kwargs):
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


def make_fc_tasks(index: Index,
                  config: dict,
                  query: dict,
                  **kwargs):
    input_type = config['nbar_dataset_type']
    output_type = config['fc_dataset_type']

    workflow = GridWorkflow(index, output_type.grid_spec)
    tiles_in = workflow.list_tiles(product=input_type.name, **query)
    _LOG.info(f"{len(tiles_in)} {input_type.name} tiles in {repr(query)}")
    tiles_out = workflow.list_tiles(product=output_type.name, **query)
    _LOG.info(f"{len(tiles_out)} {output_type.name} tiles in {repr(query)}")

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


def process_result(index: Index, result):
    for dataset in result.values:
        index.datasets.add(dataset, sources_policy='skip')
        _LOG.info('Dataset %s added at %s', dataset.id, dataset.uris)


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
@click.option('--year', 'time_range',
              callback=task_app.validate_year,
              help='Limit the process to a particular year')
@tag_option
@task_app.app_config_option
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def submit(index: Index,
           app_config: str,
           project: str,
           queue: str,
           time_range: Tuple[datetime, datetime],
           tag: str):
    _LOG.info('Tag: %s', tag)

    qsub = QSubLauncher(norm_qsub_params(
        {'project': project,
         'queue': queue,
         'name': 'fc-generate-{}'.format(tag),
         'mem': '4G',
         'noask': True,
         'wd': True,
         'ncpus': 1,
         'walltime': '1h'}))

    task_datetime = datetime.utcnow().replace(tzinfo=tz.tzutc())

    app_config_path = Path(app_config).resolve()
    app_config = paths.read_document(app_config_path)

    output_type = app_config['output_type']
    source_type = app_config['source_type']
    work_path = paths.get_product_work_directory(
        output_product=output_type,
        time=task_datetime
    )

    task_description = TaskDescription(
        type_="fc",
        task_dt=task_datetime,
        events_path=work_path.joinpath('events'),
        logs_path=work_path.joinpath('logs'),
        parameters=DefaultJobParameters(
            # TODO: Use @datacube.ui.click.parsed_search_expressions to allow params other than time from the cli?
            query=Query(index=index, time=time_range).search_terms,
            source_types=[source_type],
            output_types=[output_type],
        ),
        # Task-app framework
        runtime_state=TaskAppState(
            config_path=app_config_path,
            task_serialisation_path=work_path.joinpath('generated-tasks.pickle'),
        )
    )

    work_path.mkdir(parents=True, exist_ok=False)
    task_description.logs_path.mkdir(parents=True, exist_ok=False)
    task_description.events_path.mkdir(parents=True, exist_ok=False)

    task_description_path = work_path.joinpath('task-description.json')
    serialise.dump_structure(task_description_path, task_description)

    ret_code, qsub_stdout = qsub('generate',
                                 '-v', '-v',
                                 '--project', project,
                                 '--queue', queue,
                                 '--task-description', str(task_description_path),
                                 '--tag', tag)

    _LOG.info('Launched qsub: %d -> %s', ret_code, qsub_stdout)


@cli.command(help='Generate Tasks into file and Queue PBS job to process them')
@click.option('--project', '-P', default='u46')
@click.option('--queue', '-q', default='normal',
              type=click.Choice(['normal', 'express']))
@click.option('--no-qsub', is_flag=True, default=False, help="Skip submitting qsub for next step")
@click.option(
    '--task-description', 'task_description_file', help='',
    required=True,
    type=click.Path(exists=True, readable=True, writable=False, dir_okay=False)
)
@tag_option
@ui.verbose_option
@ui.log_queries_option
@ui.pass_index(app_name=APP_NAME)
def generate(index,
             project,
             queue,
             task_description_file,
             no_qsub,
             tag):
    task_description = _read_task_description(task_description_file)
    task_time: datetime = task_description.task_dt

    app_config = task_description.runtime_state.config_path
    config = paths.read_document(app_config)
    config['task_timestamp'] = int(task_time.timestamp())
    # TODO: This is only recording the name, not the path?
    config['app_config_file'] = Path(app_config).name

    config = make_fc_config(index, config)
    tasks = make_fc_tasks(index, config, query=task_description.parameters.query)

    num_tasks_saved = task_app.save_tasks(
        config, tasks,
        task_description.runtime_state.task_serialisation_path
    )

    _LOG.info('Tag: %s', tag)
    _LOG.info('Found %d tasks', num_tasks_saved)
    if not num_tasks_saved:
        _LOG.info("No tasks. Finishing.")
        sys.exit(0)

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
         # TODO: Add stdout/stderr from log_path
         }))

    ret_code, qsub_stdout = qsub('run',
                                 '-v', '-v',
                                 '--task-description', str(task_description_file),
                                 '--celery', 'pbs-launch',
                                 '--tag', tag)

    _LOG.info('Launched qsub: %d -> %s', ret_code, qsub_stdout)
    return ret_code


def _read_task_description(task_description_file: Path) -> TaskDescription:
    return serialise.load_structure(task_description_file, TaskDescription)


@cli.command(help='Actually process generated task file')
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option(
    '--task-description', 'task_description_file', help='',
    required=True,
    type=click.Path(exists=True, readable=True, writable=False, dir_okay=False)
)
@with_qsub_runner()
@task_app.load_tasks_option
@tag_option
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def run(index,
        dry_run,
        tag,
        task_description_file: str,
        qsub: QSubLauncher,
        runner: TaskRunner,
        *args, **kwargs):
    _LOG.info('Starting Fractional Cover processing...')
    _LOG.info('Tag: %r', tag)

    task_description = _read_task_description(Path(task_description_file))

    config, tasks = task_app.load_tasks(task_description.runtime_state.task_serialisation_path)

    if dry_run:
        task_app.check_existing_files((task['filename'] for task in tasks))
        return 0

    task_func = partial(do_fc_task, config)
    process_func = partial(process_result, index)

    try:
        runner(task_description, tasks, task_func, process_func)
        _LOG.info("Runner finished normally, triggering shutdown.")
    finally:
        runner.stop()


if __name__ == "__main__":
    cli()
