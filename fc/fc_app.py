"""
Entry point for producing Fractional Cover products.

Specifically intended for running in the PBS job queue system at the NCI.

The three entry points are:
1. datacube-fc submit
2. datacube-fc generate
3. datacube-fc run
"""
import errno
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from functools import partial
from math import ceil
from pathlib import Path
from time import time as time_now
from typing import Tuple

import click
from pandas import to_datetime

from datacube.api.grid_workflow import GridWorkflow
from datacube.api.query import Query
from datacube.index._api import Index
from datacube.model import DatasetType, GeoPolygon
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.ui import click as ui
from datacube.ui import task_app
from datacube.utils import unsqueeze_dataset
from digitalearthau import paths, serialise
from digitalearthau.qsub import QSubLauncher, with_qsub_runner, TaskRunner
from digitalearthau.runners.model import TaskDescription
from digitalearthau.runners.util import submit_subjob, init_task_app
from fc import __version__
from fc.fractional_cover import fractional_cover

APP_NAME = 'datacube-fc'
_LOG = logging.getLogger(__file__)
CONFIG_DIR = Path(__file__).parent / 'config'


def make_fc_config(index: Index, config, dry_run=False, **kwargs):
    source_product = index.products.get_by_name(config['source_product'])
    if not source_product:
        _LOG.error("Source Product %s does not exist", config['source_product'])
        return 1

    output_product_definition = deepcopy(source_product.definition)
    output_product_definition['name'] = config['output_product']
    output_product_definition['managed'] = True
    output_product_definition['description'] = config['description']
    output_product_definition['metadata']['format'] = {'name': 'NetCDF'}
    output_product_definition['metadata']['product_type'] = config.get('product_type', 'fractional_cover')

    output_product_definition['storage'] = {k: v for (k, v) in config['storage'].items()
                                            if k in ('crs', 'tile_size', 'resolution', 'origin')}

    var_def_keys = {'name', 'dtype', 'nodata', 'units', 'aliases', 'spectral_definition', 'flags_definition'}

    output_product_definition['measurements'] = [{k: v for k, v in measurement.items() if k in var_def_keys}
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

    output_product = DatasetType(source_product.metadata_type, output_product_definition)

    if not dry_run:
        _LOG.info('Created DatasetType %s', output_product.name)
        output_product = index.products.add(output_product)

    if not os.access(config['location'], os.W_OK):
        _LOG.warning('Current user appears not have write access output location: %s', config['location'])

    config['nbar_product'] = source_product
    config['fc_product'] = output_product

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
    input_product = config['nbar_product']
    output_product = config['fc_product']

    workflow = GridWorkflow(index, output_product.grid_spec)
    tiles_in = workflow.list_tiles(product=input_product.name, **query)
    _LOG.info(f"{len(tiles_in)} {input_product.name} tiles in {repr(query)}")
    tiles_out = workflow.list_tiles(product=output_product.name, **query)
    _LOG.info(f"{len(tiles_out)} {output_product.name} tiles in {repr(query)}")

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
                'name': APP_NAME,
                'version': __version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/fc.git',
                'parameters': {'configuration_file': str(config['app_config_file')}
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
    output_product = config['fc_product']

    if file_path.exists():
        raise OSError(errno.EEXIST, 'Output file already exists', str(file_path))

    nbar_tile = task['nbar']
    nbar = GridWorkflow.load(nbar_tile, measurements)

    output_measurements = config['fc_product'].measurements.values()
    fc_dataset = make_fc_tile(nbar, output_measurements, config.get('sensor_regression_coefficients'))

    def _make_dataset(labels, sources):
        assert sources
        dataset = make_dataset(product=output_product,
                               sources=sources,
                               extent=nbar.geobox.extent,
                               center_time=labels['time'],
                               uri=file_path.absolute().as_uri(),
                               app_info=get_app_metadata(config),
                               valid_data=GeoPolygon.from_sources_extents(sources, nbar.geobox))
        return dataset

    datasets = xr_apply(nbar_tile.sources, _make_dataset, dtype='O')
    fc_dataset['dataset'] = datasets_to_doc(datasets)

    write_dataset_to_netcdf(
        dataset=fc_dataset,
        filename=file_path,
        global_attributes=global_attributes,
        variable_params=variable_params,
    )
    return datasets


def process_result(index: Index, result):
    for dataset in result.values:
        index.datasets.add(dataset, sources_policy='skip')
        _LOG.info('Dataset %s added at %s', dataset.id, dataset.uris)


# pylint: disable=invalid-name
tag_option = click.option('--tag', type=str,
                          default='notset',
                          help='Unique id for the job')


@click.group(help='Datacube Fractional Cover')
@click.version_option(version=__version__)
def cli():
    pass


@cli.command(name='list', help='List installed Fractional Cover config files')
def list_configs():
    for cfg in CONFIG_DIR.glob('*.yaml'):
        click.echo(cfg)


@cli.command(help='Add FC product definitions to the datacube')
@click.argument(
    'app-config-files',
    nargs=-1,
    type=click.Path(exists=True, readable=True, writable=False, dir_okay=False)
)
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def load_config(index, app_config_files):
    for app_config_file in app_config_files:
        click.secho(f"Loading {app_config_file}", bold=True)
        app_config = paths.read_document(app_config_file)
        make_fc_config(index, app_config)


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
@click.option('--no-qsub', is_flag=True, default=False,
              help="Skip submitting job")
@tag_option
@task_app.app_config_option
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def submit(index: Index,
           app_config: str,
           project: str,
           queue: str,
           no_qsub: bool,
           time_range: Tuple[datetime, datetime],
           tag: str):
    _LOG.info('Tag: %s', tag)

    app_config_path = Path(app_config).resolve()
    app_config = paths.read_document(app_config_path)

    task_desc, task_path = init_task_app(
        job_type="fc",
        source_products=[app_config['source_product']],
        output_products=[app_config['output_product']],
        # TODO: Use @datacube.ui.click.parsed_search_expressions to allow params other than time from the cli?
        datacube_query_args=Query(index=index, time=time_range).search_terms,
        app_config_path=app_config_path,
        pbs_project=project,
        pbs_queue=queue
    )
    _LOG.info("Created task description: %s", task_path)

    if no_qsub:
        _LOG.info('Skipping submission due to --no-qsub')
        return 0

    submit_subjob(
        name='generate',
        task_desc=task_desc,
        command=[
            'generate', '-v', '-v',
            '--task-desc', str(task_path),
            '--tag', tag
        ],
        qsub_params=dict(
            mem='20G',
            wd=True,
            ncpus=1,
            walltime='1h',
            name='fc-generate-{}'.format(tag)
        )
    )


@cli.command(help='Generate Tasks into file and Queue PBS job to process them')
@click.option('--no-qsub', is_flag=True, default=False, help="Skip submitting qsub for next step")
@click.option(
    '--task-desc', 'task_desc_file', help='Task environment description file',
    required=True,
    type=click.Path(exists=True, readable=True, writable=False, dir_okay=False)
)
@tag_option
@ui.verbose_option
@ui.log_queries_option
@ui.pass_index(app_name=APP_NAME)
def generate(index: Index,
             task_desc_file: str,
             no_qsub: bool,
             tag: str):
    _LOG.info('Tag: %s', tag)

    config, task_desc = _make_config_and_description(index, Path(task_desc_file))

    num_tasks_saved = task_app.save_tasks(
        config,
        make_fc_tasks(index, config, query=task_desc.parameters.query),
        task_desc.runtime_state.task_serialisation_path
    )
    _LOG.info('Found %d tasks', num_tasks_saved)

    if not num_tasks_saved:
        _LOG.info("No tasks. Finishing.")
        sys.exit(0)

    nodes, walltime = estimate_job_size(num_tasks_saved)
    _LOG.info('Will request %d nodes and %s time', nodes, walltime)

    if no_qsub:
        _LOG.info('Skipping submission due to --no-qsub')
        return 0

    submit_subjob(
        name='run',
        task_desc=task_desc,

        command=[
            'run',
            '-vv',
            '--task-desc', str(task_desc_file),
            '--celery', 'pbs-launch',
            '--tag', tag,
        ],
        qsub_params=dict(
            name='fc-run-{}'.format(tag),
            mem='small',
            wd=True,
            nodes=nodes,
            walltime=walltime
        ),
    )


def _make_config_and_description(index: Index, task_desc_path: Path) -> Tuple[dict, TaskDescription]:
    task_desc = serialise.load_structure(task_desc_path, TaskDescription)

    task_time: datetime = task_desc.task_dt
    app_config = task_desc.runtime_state.config_path

    config = paths.read_document(app_config)

    # TODO: This carries over the old behaviour of each load. Should probably be replaced with *tag*
    config['task_timestamp'] = int(task_time.timestamp())
    config['app_config_file'] = Path(app_config)
    config = make_fc_config(index, config)

    return config, task_desc


@cli.command(help='Actually process generated task file')
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option(
    '--task-desc', 'task_desc_file', help='Task environment description file',
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
        dry_run: bool,
        tag: str,
        task_desc_file: str,
        qsub: QSubLauncher,
        runner: TaskRunner,
        *args, **kwargs):
    _LOG.info('Starting Fractional Cover processing...')
    _LOG.info('Tag: %r', tag)

    task_desc = serialise.load_structure(Path(task_desc_file), TaskDescription)
    config, tasks = task_app.load_tasks(task_desc.runtime_state.task_serialisation_path)

    if dry_run:
        task_app.check_existing_files((task['filename'] for task in tasks))
        return 0

    task_func = partial(do_fc_task, config)
    process_func = partial(process_result, index)

    try:
        runner(task_desc, tasks, task_func, process_func)
        _LOG.info("Runner finished normally, triggering shutdown.")
    finally:
        runner.stop()


if __name__ == "__main__":
    cli()
