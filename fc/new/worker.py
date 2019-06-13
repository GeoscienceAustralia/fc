"""

TODO:
- [ ] Logging
- [ ] Error Handling

"""
import yaml
from io import StringIO
from types import SimpleNamespace

import datacube
from datacube.model.utils import make_dataset
from datacube.virtual import construct, Transformation
from fc.fc_app import _get_app_metadata, dataset_to_geotif_yaml, calc_uris


class NRTJob:
    # Database Configuration
    db_hostname = None
    db_port = None
    db_username = None
    db_password = None

    # SQS config for finding work
    sqs_queue_url = None
    sqs_message_prefix = None
    sqs_poll_time_sec = None
    job_max_time_sec = None
    max_job_per_worker = None

    type = None  # wofs or fc

    log_level = None

    # From SQS Message
    input_file = ''

    # Should be from SQS message but isn't
    input_s3_bucket = None

    # Output Configuration
    output_s3_bucket = None
    output_path = None
    make_s3_public = None
    bigtiff = None
    gdal_tiff_internal_mask = None
    gdal_tiff_ovr_blocksize = None

    # Hard Coded / assumed
    # - Output file name generation pattern
    # - Resolution
    # - Algorithm - WOfS

    # Looked up from DB
    # - Input Product
    # - Input Dataset YAML + Location
    # -


class Job:
    # Other
    query_pattern = ''
    output_format = ''
    output_path_pattern = ''

    input_vproduct = ''
    output_product = ''

    transformation = Transformation()

    def generate_output_filename(self):
        pass

    def generate_dataset_metadata(self, *input_datasets):
        pass


example_task = yaml.safe_load(StringIO('''
config:
    output_format: tiff
    output_path_pattern: s3://dea-public-data/foo/bar/{year}/{tile_id}/{date}-{id}.tif
    ?? transformation: ??
    ?? default metadata values: ??
    output_product_name: ls8_fc_scene

input:
    input_product: ...
    input_dataset_yaml: ...
    input_dataset_location: ...

OR

    virtual_product_box: ...
    virtual_product_definition: ...

'''))


def worker(job):
    # load configuration and setup
    with open('proto_config.yaml') as conf_file:
        config = yaml.safe_load(conf_file)
    vproduct = construct(config['input'])

    dc = datacube.Datacube()
    output_product = dc.index.products.get_by_name(config['output_product_name'])
    vdbag = vproduct.query(dc=dc, id=job.dataset_id)

    box = vproduct.group(vdbag)

    # Serialise the box into the task
    task_job = SimpleNamespace(
        box=box,

    )

    # Load and perform processing
    output_data = vproduct.fetch(box)

    # compute filename
    variable_params = {band: None
                       for band in vproduct.output_measurements(vdbag.product_definitions)}
    uri, band_uris = calc_uris(config['output_path_pattern'], variable_params)

    # generate dataset metadata
    input_dataset = next(iter(vdbag.pile))
    dataset = make_dataset(product=output_product,
                           sources=[input_dataset],
                           extent=box.geobox.extent,
                           center_time=input_dataset.center_time,
                           uri=uri,
                           band_uris=band_uris,
                           app_info=_get_app_metadata(config),
                           )
    # write data to disk
    dataset_to_geotif_yaml(
        dataset=output_data,
        filename=uri,
        variable_params=variable_params,
    )
    # ensure filepath
    # write

    # record dataset record to database
    # OR NOT


def run(query, transformation):
    transformer = transformation()

    for dataset in execute_query(query):
        pass


def load_data(job):
    return {}


def execute_query():
    pass
