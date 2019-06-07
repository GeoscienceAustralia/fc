"""

TODO:
- [ ] Logging
- [ ] Error Handling

"""
from io import StringIO
from typing import Dict

import xarray
import yaml

from datacube.utils import unsqueeze_dataset
from datacube.virtual import Transformation, Measurement, construct
from fc.fractional_cover import fractional_cover


class FractionalCover(Transformation):
    def __init__(self, regression_coefficients, **settings):
        self.regression_coefficients = regression_coefficients
        self._measurements = {
            'BS': Measurement(name='BS', dtype='int8', nodata=-1, units='percent'),
            'PV': Measurement(name='PV', dtype='int8', nodata=-1, units='percent'),
            'NPV': Measurement(name='NPV', dtype='int8', nodata=-1, units='percent'),
            'UE': Measurement(name='UE', dtype='int8', nodata=-1, units='percent'),
        }

    def compute(self, data):
        input_tile = data.squeeze('time').drop('time')
        data = fractional_cover(input_tile, self.measurements, self.regression_coefficients)
        output_tile = unsqueeze_dataset(data, 'time', data.time.values[0])
        return output_tile

    def measurements(self, input_measurements):
        return self._measurements


class Transformation:
    def __init__(self, **settings):
        """ Initialize the transformation object with the given settings. """

    def compute(self,
                data: xarray.Dataset) -> xarray.Dataset:
        return data

    def measurements(self,
                     input_measurements: Dict[str, Measurement]) -> Dict[str, Measurement]:
        return input_measurements


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


class NCIJob:
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

input:
    input_product: ...
    input_dataset_yaml: ...
    input_dataset_location: ...

OR

    virtual_product_definition: ...

'''))


def worker(job, transformation):
    # load configuration and setup
    with open('proto_config.yaml') as conf_file:
        config = yaml.safe_load(conf_file)
    vproduct = construct(config['input'])

    # Load and compute data
    vdbag = vproduct.search(id=job.dataset_id)

    box = vproduct.group(vdbag)

    data = vproduct.fetch(box)

    # generate dataset metadata
    input_dataset = next(iter(vdbag.pile))

    # write data to disk
        # compute filename
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
