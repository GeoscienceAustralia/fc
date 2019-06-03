"""
Entry point for producing Fractional Cover products.

Specifically intended for running in the PBS job queue system at the NCI.

Following cli commands are supported:
1. datacube-fc list
2. datacube-fc ensure-products
3. datacube-fc submit
4. datacube-fc generate
5. datacube-fc run
"""
import logging
import os
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from time import time as time_now
from typing import Tuple, Union, Iterable

import click
import xarray
from boltons import fileutils
from math import ceil
from pandas import to_datetime

from datacube.api.grid_workflow import GridWorkflow, Tile
from datacube.api.query import Query
from datacube.index._api import Index
from datacube.model import DatasetType
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc

try:
    from datacube.drivers.netcdf import write_dataset_to_netcdf
except ImportError:
    from datacube.storage.storage import write_dataset_to_netcdf
from datacube.helpers import write_geotiff
from datacube.ui import click as ui
from datacube.utils import geometry
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


def polygon_from_sources_extents(sources, geobox):
    sources_union = geometry.unary_union(source.extent.to_crs(geobox.crs) for source in sources)
    valid_data = geobox.extent.intersection(sources_union)
    resolution = min([abs(x) for x in geobox.resolution])
    return valid_data.simplify(tolerance=resolution * 0.01)


def _make_fc_config(index: Index, config: dict, dry_run):
    """
    Refine output fc configuration file. Before returning the updated file,
    ensure that the products exist for the given FC config.
    """
    if not os.path.exists(config['location']):
        os.makedirs(config['location'])
    elif not os.access(config['location'], os.W_OK):
        _LOG.warning('Current user appears not have write access output location: %s', config['location'])

    source_product, output_product = _ensure_products(config, index, dry_run)

    # The input config has `source_product` and `output_product` fields which are names. Perhaps these should
    # just replace them?
    config['nbart_product'] = source_product
    config['fc_product'] = output_product

    config['variable_params'] = _build_variable_params(config)

    if 'task_timestamp' not in config:
        config['task_timestamp'] = int(time_now())

    return config


_MEASUREMENT_KEYS_TO_COPY = ('zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'attrs')


def _build_variable_params(config: dict) -> dict:
    chunking = config['storage']['chunking']
    chunking = [chunking[dim] for dim in config['storage']['dimension_order']]

    variable_params = {}
    for mapping in config['measurements']:
        measurement_name = mapping['name']
        variable_params[measurement_name] = {
            k: v
            for k, v in mapping.items()
            if k in _MEASUREMENT_KEYS_TO_COPY
        }
        variable_params[measurement_name]['chunksizes'] = chunking
    return variable_params


def _ensure_products(app_config: dict, index: Index, dry_run: bool) -> Tuple[DatasetType, DatasetType]:
    source_product_name = app_config['source_product']
    source_product = index.products.get_by_name(source_product_name)
    if not source_product:
        raise ValueError(f"Source product {source_product_name} does not exist")

    output_product = DatasetType(
        source_product.metadata_type,
        _create_output_definition(app_config, source_product)
    )
    if not dry_run:
        _LOG.info('Add the output product definition for %s in the database.', output_product.name)
        output_product = index.products.add(output_product)
    return source_product, output_product


def _create_output_definition(config: dict, source_product: DatasetType) -> dict:
    output_product_definition = deepcopy(source_product.definition)
    output_product_definition['name'] = config['output_product']
    output_product_definition['managed'] = True
    output_product_definition['description'] = config['description']
    output_product_definition['metadata']['format'] = {'name': config['storage']['driver']}
    output_product_definition['metadata']['product_type'] = config.get('product_type', 'fractional_cover')
    output_product_definition['storage'] = {
        k: v for (k, v) in config['storage'].items()
        if k in ('crs', 'tile_size', 'resolution', 'origin')
    }
    var_def_keys = {'name', 'dtype', 'nodata', 'units', 'aliases', 'spectral_definition', 'flags_definition'}

    output_product_definition['measurements'] = [
        {k: v for k, v in measurement.items() if k in var_def_keys}
        for measurement in config['measurements']
    ]
    # Validate the output product definition
    DatasetType.validate(output_product_definition)
    return output_product_definition


def _get_filename(config, tile_index, sources):
    file_path_template = str(Path(config['location'], config['file_path_template']))
    return file_path_template.format(tile_index=tile_index,
                                     start_time=to_datetime(sources.time.values[0]).strftime('%Y%m%d%H%M%S%f'),
                                     end_time=to_datetime(sources.time.values[-1]).strftime('%Y%m%d%H%M%S%f'),
                                     version=config['task_timestamp'])


def _make_fc_tasks(index: Index,
                   config: dict,
                   query: dict):
    """
    Generate an iterable of 'tasks', matching the provided filter parameters.
    Tasks can be generated for:
    - all of time
    - 1 particular year
    - a range of years
    """
    input_product = config['nbart_product']
    output_product = config['fc_product']

    workflow = GridWorkflow(index, output_product.grid_spec)

    tiles_in = workflow.list_tiles(product=input_product.name, **query)
    # _LOG.info(f"{len(tiles_in)} {input_product.name} tiles in {repr(query)}")
    _LOG.info('%d %s tiles in %s', len(tiles_in), input_product.name, str(repr(query)))
    # _LOG.info('Found %d tasks', num_tasks_saved)
    tiles_out = workflow.list_tiles(product=output_product.name, **query)
    # _LOG.info(f"{len(tiles_out)} {output_product.name} tiles in {repr(query)}")
    _LOG.info('%d %s tiles in %s', len(tiles_out), output_product.name, str(repr(query)))
    return (
        dict(
            nbart=workflow.update_tile_lineage(tile),
            tile_index=key,
            filename=_get_filename(config, tile_index=key, sources=tile.sources)
        )
        for key, tile in tiles_in.items() if key not in tiles_out
    )


def _get_app_metadata(config):
    doc = {
        'lineage': {
            'algorithm': {
                'name': APP_NAME,
                'version': __version__,
                'repo_url': 'https://github.com/GeoscienceAustralia/fc.git',
                'parameters': {'configuration_file': str(config['app_config_file'])}
            },
        }
    }
    return doc


def _make_fc_tile(nbart: xarray.Dataset, measurements, regression_coefficients):
    input_tile = nbart.squeeze('time').drop('time')
    data = fractional_cover(input_tile, measurements, regression_coefficients)
    output_tile = unsqueeze_dataset(data, 'time', nbart.time.values[0])
    return output_tile


def calc_uris(file_path, variable_params):
    base, ext = os.path.splitext(file_path)
    if ext == '.tif':
        # the file_path value used is highly coupled to
        # dataset_to_geotif_yaml since it's assuming the
        # yaml file is in the same dir as the tif file
        abs_paths, rel_files, yml = tif_filenames(file_path, variable_params.keys())
        uri = yml.as_uri()
        band_uris = {band: {'path': uri, 'layer': band} for band, uri in rel_files.items()}
        if all_files_exist(abs_paths.values()):
            raise FileExistsError('All output files already exist ', str(list(rel_files.values())))
    else:
        band_uris = None
        uri = file_path.absolute().as_uri()
        if file_path.exists():
            raise FileExistsError('Output file already exists', str(file_path))

    return uri, band_uris


def _do_fc_task(config, task):
    """
    Load data, run FC algorithm, attach metadata, and write output.
    :param dict config: Config object
    :param dict task: Dictionary of tasks
    :return: Dataset objects representing the generated data that can be added to the index
    :rtype: list(datacube.model.Dataset)
    """

    global_attributes = config['global_attributes']
    variable_params = config['variable_params']
    file_path = Path(task['filename'])
    output_product = config['fc_product']

    uri, band_uris = calc_uris(file_path, variable_params)

    nbart_tile: Tile = task['nbart']
    nbart = GridWorkflow.load(nbart_tile, ['green', 'red', 'nir', 'swir1', 'swir2'])

    output_measurements = config['fc_product'].measurements.values()
    fc_dataset = _make_fc_tile(nbart, output_measurements, config.get('sensor_regression_coefficients'))

    def _make_dataset(labels, sources):
        assert sources
        dataset = make_dataset(product=output_product,
                               sources=sources,
                               extent=nbart.geobox.extent,
                               center_time=labels['time'],
                               uri=uri,
                               band_uris=band_uris,
                               app_info=_get_app_metadata(config),
                               valid_data=polygon_from_sources_extents(sources, nbart.geobox))
        return dataset

    datasets = xr_apply(nbart_tile.sources, _make_dataset, dtype='O')
    fc_dataset['dataset'] = datasets_to_doc(datasets)

    base, ext = os.path.splitext(file_path)
    if ext == '.tif':
        dataset_to_geotif_yaml(
            dataset=fc_dataset,
            filename=file_path,
            variable_params=variable_params,
        )
    else:
        write_dataset_to_netcdf(
            dataset=fc_dataset,
            filename=file_path,
            global_attributes=global_attributes,
            variable_params=variable_params,
        )

    return datasets


def _process_result(index: Index, result):
    for dataset in result.values:
        index.datasets.add(dataset, sources_policy='skip')
        _LOG.info('Dataset %s added at %s', dataset.id, dataset.uris)


# pylint: disable=invalid-name
tag_option = click.option('--tag', type=str,
                          default='notset',
                          help='Unique id for the job')

# pylint: disable=invalid-name
pbs_email_options = click.option('--email-options', '-m', default='abe',
                                 type=click.Choice(['a', 'b', 'e', 'n', 'ae', 'ab', 'be', 'abe']),
                                 help='Send Email when execution is, \n'
                                      '[a = aborted | b = begins | e = ends | n = do not send email]')

# pylint: disable=invalid-name
pbs_email_id = click.option('--email-id', '-M', default='nci.monitor@dea.ga.gov.au',
                            help='Email Recipient List')


@click.group(help='Datacube Fractional Cover')
@click.version_option(version=__version__)
def cli():
    """
    Instantiate a click 'Datacube fractional cover' group object to register the following sub-commands for
    different bits of FC processing:
         1) list
         2) ensure-products
         3) submit
         4) generate
         5) run
    :return: None
    """
    pass


@cli.command(name='list', help='List installed Fractional Cover config files')
def list_configs():
    """
     List installed FC config files
    :return: None
    """
    for cfg in CONFIG_DIR.glob('*.yaml'):
        click.echo(cfg)


@cli.command(name='ensure-products',
             help="Ensure the products exist for the given FC config, creating them if necessary.")
@click.option('--app-config', help='App configuration file',
              type=click.Path(exists=True, readable=True, writable=False, dir_okay=False),
              required=True)
@click.option('--dry-run', is_flag=True, default=False,
              help='Check product definition without modifying the database')
@ui.config_option
@ui.verbose_option
@ui.pass_index(app_name=APP_NAME)
def ensure_products(index, app_config, dry_run):
    """
    Ensure the products exist for the given FC config, creating them if necessary.
    If dry run is disabled, the validated output product definition will be added to the database.
    """
    # TODO: Add more validation of config?
    click.secho(f"Loading {app_config}", bold=True)
    app_config_file = paths.read_document(app_config)
    _, out_product = _ensure_products(app_config_file, index, dry_run)
    click.secho(f"Output product definition for {out_product.name} product exits in the database for the given "
                f"FC input config file")


def _estimate_job_size(num_tasks):
    """ Translate num_tasks to number of nodes and walltime
    """
    max_nodes = 20
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
@pbs_email_options
@pbs_email_id
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option('--local', is_flag=True, default=False, help='Experimental. Run the tasks locally; not on qsub.')
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
           tag: str,
           email_options: str,
           email_id: str,
           dry_run: bool,
           local: bool):
    """
    Kick off two stage PBS job

    Stage 1 (Generate task file):
        The task-app machinery loads a config file, from a path specified on the
        command line, into a dict.

        If dry is enabled, a dummy DatasetType is created for tasks generation without indexing
        the product in the database.
        If dry run is disabled, generate tasks into file and queue PBS job to process them.

    Stage 2 (Run):
        During normal run, following are performed:
           1) Tasks shall be yielded for dispatch to workers.
           2) Load data
           3) Run FC algorithm
           4) Attach metadata
           5) Write output files and
           6) Finally index the newly created FC output netCDF files

        If dry run is enabled, application only prepares a list of output files to be created and does not
        record anything in the database.
    """

    submit_command(index,
                   app_config,
                   project,
                   queue,
                   no_qsub,
                   time_range,
                   tag,
                   email_options,
                   email_id,
                   dry_run,
                   local)
    return 0


def submit_command(index: Index,
                   app_config: str,
                   project: str,
                   queue: str,
                   no_qsub: bool,
                   time_range: Tuple[datetime, datetime],
                   tag: str,
                   email_options: str,
                   email_id: str,
                   dry_run: bool,
                   local: bool):
    """
    Kick off a two stage PBS job.

    :return: Created task description
    """
    _LOG.info('Tag: %s', tag)

    app_config_path = Path(app_config).resolve()
    app_config = paths.read_document(app_config_path)

    if not time_range or not all(time_range):
        query_args = Query(index=index).search_terms
    else:
        query_args = Query(index=index, time=time_range).search_terms

    task_desc, task_path = init_task_app(
        job_type="fc",
        source_products=[app_config['source_product']],
        output_products=[app_config['output_product']],
        # TODO: Use @datacube.ui.click.parsed_search_expressions to allow params other than time from the cli?
        datacube_query_args=query_args,
        app_config_path=app_config_path,
        pbs_project=project,
        pbs_queue=queue
    )
    _LOG.info("Created task description: %s", task_path)

    if no_qsub:
        _LOG.info('Skipping submission due to --no-qsub')
        # currently just used for teting
        return task_path

    # If dry run is not enabled just pass verbose option
    dry_run_option = '--dry-run' if dry_run else '-v'
    extra_qsub_args = '-M {0} -m {1}'.format(email_id, email_options)

    # Append email options and email id to the PbsParameters dict key, extra_qsub_args
    task_desc.runtime_state.pbs_parameters.extra_qsub_args.extend(extra_qsub_args.split(' '))

    if not local:
        submit_subjob(
            name='generate',
            task_desc=task_desc,
            command=[
                'generate', '-vv',
                '--task-desc', str(task_path),
                '--tag', tag,
                '--log-queries',
                '--email-id', email_id,
                '--email-options', email_options,
                dry_run_option,
            ],
            qsub_params=dict(
                name='fc-generate-{}'.format(tag),
                mem='medium',
                wd=True,
                nodes=1,
                walltime='1h'
            )
        )
    else:
        _LOG.info("local task execution! WARNING: This has only been tested for 1 or 2 task jobs.")
        generate_command(index=index,
                         task_desc_file=str(task_path),
                         tag=tag,
                         no_qsub=False,
                         email_options=email_options,
                         email_id=email_id,
                         dry_run=dry_run,
                         local=local)
    return 0


@cli.command(help='Generate Tasks into file and Queue PBS job to process them')
@click.option('--no-qsub', is_flag=True, default=False, help="Skip submitting qsub for next step")
@click.option('--task-desc', 'task_desc_file', help='Task environment description file',
              required=True,
              type=click.Path(exists=True, readable=True, writable=False, dir_okay=False))
@tag_option
@pbs_email_options
@pbs_email_id
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option('--local', is_flag=True, default=False, help='Experimental. Run the tasks locally; not on qsub.')
@ui.verbose_option
@ui.log_queries_option
@ui.pass_index(app_name=APP_NAME)
def generate(index: Index,
             task_desc_file: str,
             no_qsub: bool,
             tag: str,
             email_options: str,
             email_id: str,
             dry_run: bool,
             local: bool):
    """
    Generate Tasks into file and Queue PBS job to process them.

    If dry run is enabled, do not add the new products to the database.
    """
    return generate_command(index,
                            task_desc_file,
                            no_qsub,
                            tag,
                            email_options,
                            email_id,
                            dry_run,
                            local)


def generate_command(index: Index,
                     task_desc_file: str,
                     no_qsub: bool,
                     tag: str,
                     email_options: str,
                     email_id: str,
                     dry_run: bool,
                     local: bool):
    _LOG.info('Tag: %s', tag)

    config, task_desc = _make_config_and_description(index, Path(task_desc_file), dry_run)

    num_tasks_saved = task_app.save_tasks(
        config,
        _make_fc_tasks(index, config, query=task_desc.parameters.query),
        task_desc.runtime_state.task_serialisation_path
    )
    _LOG.info('Found %d tasks', num_tasks_saved)

    if not num_tasks_saved:
        _LOG.info("No tasks. Finishing.")
        return 0

    nodes, walltime = _estimate_job_size(num_tasks_saved)
    _LOG.info('Will request %d nodes and %s time', nodes, walltime)

    if no_qsub:
        _LOG.info('Skipping submission due to --no-qsub')
        return 0

    # If dry run is not enabled just pass verbose option
    dry_run_option = '--dry-run' if dry_run else '-v'
    extra_qsub_args = '-M {0} -m {1}'.format(email_id, email_options)

    # Append email options and email id to the PbsParameters dict key, extra_qsub_args
    task_desc.runtime_state.pbs_parameters.extra_qsub_args.extend(extra_qsub_args.split(' '))

    if not local:
        submit_subjob(
            name='run',
            task_desc=task_desc,
            command=[
                'run',
                '-vv',
                '--task-desc', str(task_desc_file),
                '--celery', 'pbs-launch',
                '--tag', tag,
                dry_run_option,
            ],
            qsub_params=dict(
                name='fc-run-{}'.format(tag),
                mem='medium',
                wd=True,
                nodes=nodes,
                walltime=walltime
            ),
        )
    else:
        _LOG.info("local task execution! WARNING: This has only been tested for 1 or 2 task jobs.")
        runner = TaskRunner()
        run_command(index,
                    dry_run=dry_run,
                    tag=tag,
                    task_desc_file=str(task_desc_file),
                    runner=runner)
    return 0



def _make_config_and_description(index: Index, task_desc_path: Path, dry_run: bool) -> Tuple[dict, TaskDescription]:
    task_desc = serialise.load_structure(task_desc_path, TaskDescription)

    task_time: datetime = task_desc.task_dt
    app_config = task_desc.runtime_state.config_path

    config = paths.read_document(app_config)

    # TODO: This carries over the old behaviour of each load. Should probably be replaced with *tag*
    config['task_timestamp'] = int(task_time.timestamp())
    config['app_config_file'] = Path(app_config)
    config = _make_fc_config(index, config, dry_run)

    return config, task_desc


@cli.command(help='Process generated task file')
@click.option('--dry-run', is_flag=True, default=False, help='Check if output files already exist')
@click.option('--task-desc', 'task_desc_file', help='Task environment description file',
              required=True,
              type=click.Path(exists=True, readable=True, writable=False, dir_okay=False))
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
    """
    Process generated task file. If dry run is enabled, only check for the existing files
    """
    return run_command(index, dry_run, tag, task_desc_file, runner)


def run_command(index,
                dry_run: bool,
                tag: str,
                task_desc_file: str,
                runner: TaskRunner):
    task_desc = serialise.load_structure(Path(task_desc_file), TaskDescription)
    config, tasks = task_app.load_tasks(task_desc.runtime_state.task_serialisation_path)

    if dry_run:
        _LOG.info('Starting Fractional Cover Dry Run...')
        task_app.check_existing_files((task['filename'] for task in tasks))
        return 0

    _LOG.info('Starting Fractional Cover processing...')
    _LOG.info('Tag: %r', tag)
    task_func = partial(_do_fc_task, config)
    process_func = partial(_process_result, index)

    try:
        runner(task_desc, tasks, task_func, process_func)
        _LOG.info("Runner finished normally, triggering shutdown.")
    finally:
        runner.stop()
    return 0


def all_files_exist(filenames: Iterable):
    """
    Return True if all files in a list exist.

    :param filenames: A list of file paths.
    :return:
    """
    isthere = (os.path.isfile(i) for i in filenames)
    return all(isthere)


def tif_filenames(filename: Union[Path, str], bands: list, sep='_'):
    """
    Turn one file name into several file names, one per band.
    This turns a .tif filename into two dictionaries of filenames,
    For abs and rel the band as the key, with the band inserted into the file names.
        i.e ls8_fc.tif -> ls8_fc_BS.tif  (Last underscore is separator)
    The paths in abs_paths are absolute
    The paths in rel_files are relative to the yml
    yml is the path location to where the yml file will be written

    :param filename: a Path.
    :param bands: a list of bands/measurements
    :param sep: the separator between the base name and the band.
    :return: (abs_paths, rel_files, yml)
    """
    base, ext = os.path.splitext(filename)
    assert ext == '.tif'
    yml = Path(base + '.yml').absolute()
    abs_paths = {}
    rel_files = {}
    for band in bands:
        build = Path(base + sep + band + ext)
        abs_paths[band] = build.absolute().as_uri()
        # This is to get relative paths
        rel_files[band] = os.path.basename(build)
    return abs_paths, rel_files, yml


def dataset_to_geotif_yaml(dataset: xarray.Dataset,
                           filename: Union[Path, str],
                           variable_params=None):
    """
    Write the dataset out as a set of geotifs with metadata in a yaml file.
    There will be one geotiff file per band.
    The band name is added into the file name.
    i.e ls8_fc.tif -> ls8_fc_BS.tif

    :param dataset:
    :param filename: Output filename
    :param variable_params: dict of variable_name: {param_name: param_value, [...]}
                            Used to get band names.

    """

    bands = variable_params.keys()
    abs_paths, _, yml = tif_filenames(filename, bands)

    Path(filename).parent.mkdir(exist_ok=True)

    # Write out the yaml file
    with fileutils.atomic_save(str(yml)) as yaml_dst:
        yaml_dst.write(dataset.data_vars['dataset'].values[0])

    # Iterate over the bands
    for key, bandfile in abs_paths.items():
        slim_dataset = dataset[[key]]  # create a one band dataset
        attrs = slim_dataset[key].attrs.copy()  # To get nodata in
        del attrs['crs']  # It's  format is poor
        del attrs['units']  # It's  format is poor
        slim_dataset[key] = dataset.data_vars[key].astype('int16', copy=True)
        write_geotiff(bandfile, slim_dataset.isel(time=0), profile_override=attrs)


if __name__ == "__main__":
    cli()
