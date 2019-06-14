"""

TODO:
- [ ] Logging
- [ ] Error Handling

"""
import click
import yaml
from io import StringIO

import datacube
from datacube.model.utils import make_dataset, datasets_to_doc
from datacube.virtual import construct, Transformation
from datacube.virtual.impl import VirtualDatasetBag
from fc.fc_app import dataset_to_geotif_yaml, calc_uris
from .dask import dask_compute_stream


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

task = {
    'virtual_product_box': None,
    'virtual_product_def': None,

    'file_output': dict(
        location='/g/data/u46/users/dsg547/sandpit/odc_testing/a_test/LS8_OLI_FC/',
        file_path_template='{x}_{y}/LS8_OLI_FC_3577_{x}_{y}_{start_time}_v{version}'
    ),

    'output_product': None,
}


# 'outputs': {
#     'BS': '/a/sdljn/asdlkn/LS8_FC_20182928_BS.tif',
#     'PV': '/a/sdljn/asdlkn/LS8_FC_20182928_PV.tif',
#     'NPV': '/a/sdljn/asdlkn/LS8_FC_20182928_NPV.tif',
#     'UE': '/a/sdljn/asdlkn/LS8_FC_20182928_UE.tif',
#     'metadata': '/a/sdljn/asdlkn/LS8_FC_20182928.yaml',
# }


@click.command
@click.option('config_file')
def main(config_file):
    with open(config_file) as conf_file:
        config = yaml.safe_load(conf_file)
    vproduct = construct(config['input'])

    dc = datacube.Datacube()
    output_product_name = config['task_generation']['output_product']
    output_product = dc.index.products.get_by_name(output_product_name)

    datasets = datasets_that_need_to_be_processed(dc.index, config['task_generation']['input_product'],
                                                  output_product_name)

    bags = map(bag_maker(output_product_name, output_product), datasets)

    boxes = map(box_maker(vproduct), bags)

    tasks = map(task_maker(config, output_product), boxes)

    completed = dask_compute_stream(client, execute_task, tasks)


def bag_maker(product_name, product):
    def dataset_to_bag(dataset):
        return VirtualDatasetBag([dataset], None, {product_name: product})

    return dataset_to_bag


def box_maker(vproduct):
    def bag_to_box(bag):
        return vproduct.group(bag)

    return bag_to_box


def task_maker(config, output_product):
    def make_task(box):
        return dict(
            box=box,
            vproduct_def=config['input'],
            file_output=config['file_output'],
            output_product=output_product,

        )

    return make_task


def execute_task(task):
    vproduct = construct(task['virtual_product_def'])
    box = task['box']

    # Load and perform processing
    output_data = vproduct.fetch(box)

    # compute base filename
    variable_params = {band: None
                       for band in vproduct.output_measurements(box.product_definitions)}
    uri, band_uris = calc_uris(task['file_output']['file_path_template'], variable_params)

    # generate dataset metadata
    input_dataset = next(iter(box.pile))
    dataset = make_dataset(product=task['output_product'],
                           sources=[input_dataset],
                           extent=box.geobox.extent,
                           center_time=input_dataset.center_time,
                           uri=uri,
                           band_uris=band_uris,
                           app_info=task['virtual_product_def'],
                           )

    output_data['dataset'] = datasets_to_doc([dataset])
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


if __name__ == '__main__':
    main()
