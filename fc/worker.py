"""

TODO:
- [ ] Logging
- [ ] Error Handling

"""
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


class Job:
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


def run(query, transformation):
    transformer = transformation()

    for dataset in execute_query(query):
        pass


def load_data(job):
    return {}


def execute_query():
    pass
