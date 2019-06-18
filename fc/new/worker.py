"""

TODO:
- [ ] Logging
- [ ] Error Handling

"""

from collections import namedtuple

import click
import yaml

import datacube
from datacube.model.utils import make_dataset, datasets_to_doc
from datacube.virtual import construct
from datacube.virtual.impl import VirtualDatasetBag
from fc.fc_app import dataset_to_geotif_yaml, calc_uris
from fc.new._dask import dask_compute_stream

Task = namedtuple('Task', ['box', 'virtual_product_def', 'file_output', 'output_product'])

task = {
    'box': None,
    'virtual_product_def': None,

    'file_output': dict(
        location='/g/data/u46/users/dsg547/sandpit/odc_testing/a_test/LS8_OLI_FC/',
        file_path_template='{x}_{y}/LS8_OLI_FC_3577_{x}_{y}_{start_time}_v{version}'
    ),

    'output_product': None,
}


@click.command()
@click.argument('config_file')
def main(config_file):
    # Load Configuration file
    with open(config_file) as conf_file:
        config = yaml.safe_load(conf_file)
    vproduct = construct(**config['virtual_product_specification'])

    # Connect to the ODC Index
    dc = datacube.Datacube()
    input_product_name = config['task_generation']['input_product']
    input_product = dc.index.products.get_by_name(input_product_name)

    # Find what needs to be processed
    datasets = dc.index.datasets.search(limit=3, product=config['task_generation']['input_product'])
    # datasets = datasets_that_need_to_be_processed(dc.index, config['task_generation']['input_product'],
    #                                               output_product_name)

    # Divide into tasks
    bags = (
        VirtualDatasetBag([dataset], None, {input_product_name: input_product})
        for dataset in datasets
    )

    boxes = map(box_maker(vproduct), bags)

    tasks = map(task_maker(config, input_product), boxes)

    tasks = list(tasks)
    print(len(tasks))
    print(tasks)

    # Execute the tasks across the dask cluster
    from dask.distributed import Client
    client = Client()
    completed = dask_compute_stream(client, execute_task, tasks)

    try:
        for result in completed:
            print(result)
    except Exception as e:
        print(e)


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
        return Task(
            box=box,
            virtual_product_def=config['virtual_product_specification'],
            file_output=config['file_output'],
            output_product=output_product,

        )

    return make_task


def execute_task(task: Task):
    vproduct = construct(**task.virtual_product_def)

    # Load and perform processing
    output_data = vproduct.fetch(task.box)

    # compute base filename
    variable_params = {band: None
                       for band in vproduct.output_measurements(box.product_definitions)}
    uri, band_uris = calc_uris(task.file_output['file_path_template'], variable_params)

    # generate dataset metadata
    input_dataset = next(iter(task.box.pile))
    dataset = make_dataset(product=task.output_product,
                           sources=[input_dataset],
                           extent=task.box.geobox.extent,
                           center_time=input_dataset.center_time,
                           uri=uri,
                           band_uris=band_uris,
                           app_info=task.virtual_product_def,
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
