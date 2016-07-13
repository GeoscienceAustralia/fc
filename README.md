# Fraction Cover (fc)

Fractional Cover measures the photosynthetic, non-photosynthetic and bare earth components of a Landsat image.

## Installation
To install the module on raijin:

Checout the branch you wish to install to temp directory

```
$ module use /g/data/v10/public/modules/modulefiles/
$ sh package-module.sh 
```
You will be promted to check the package location and version.
If it is correct, type **`y`** and press enter
```
# Packaging agdc-fc v 1.0.0 to /g/data/v10/public/modules/agdc-fc/1.0.0 #
Continue? 
```

## Running
To run fractional cover:

```
$ module use /g/data/v10/public/modules/modulefiles/
$ module load agdc-fc
$ datacube-fc-launcher list
```

This will list the availiable app configs.
```
ls5_fc_albers.yaml
ls7_fc_albers.yaml
ls8_fc_albers.yaml
```

Run the launcher with the `qsub` command, specifying the app config, year, and PBS properties:
* `-q normal` the queue to use, either normal or express
* `-P v10` project to use
* `-n 1` number of nodes
* `-t 1` number of hours (walltime)
```
datacube-fc-launcher qsub ls5_fc_albers.yaml 1994 -q normal -P v10 -n 1 -t 1
```

It will check to make sure it can access the database:
```
Version: 1.1.4
Read configurations files from: [u'/home/547/adh547/fc/datacube.conf']
Host: 130.56.244.227:6432
Database: unification
User: adh547


Attempting connect
Success.
You have MANAGE privileges.
```

Then is will create the task file, and create the output product definition in the database if needed:
```
datacube-fc -v --app-config "/g/data/v10/public/modules/agdc-fc/1.0.0/config/ls5_fc_albers_test.yaml" --year 1994 --save-tasks "/g/data2/v10/public/modules/agdc-fc/1.0.0/scripts/ls5_fc_albers_test_1993.bin"
RUN? [Y/n]:

2016-07-13 18:38:56,308 INFO Created DatasetType ls5_fc_albers
2016-07-13 18:39:01,997 INFO 291 tasks discovered
2016-07-13 18:39:01,998 INFO 291 tasks discovered
2016-07-13 18:39:02,127 INFO Saved config and tasks to /g/data2/v10/public/modules/agdc-fc/1.0.0/scripts/ls5_fc_albers_test_1993.bin
```

It will loop through every task :
```
datacube-fc -v -C "/home/547/adh547/fc/datacube.conf" --load-tasks "/g/data2/v10/public/modules/agdc-fc/1.0.0/scripts/ls5_fc_albers_test_1993.bin" --dry-run
RUN? [y/N]:

Running task: {'filename': '/g/data/u46/users/adh547/datacube/002/LS5_TM_FC/15_-39/LS5_TM_FC_3577_15_-39_19930513231246500000.nc', 'nbar': {'geobox': GeoBox(4000, 4000, Affine(25.0, 0.0, 1500000.0,
       0.0, -25.0, -3800000.0), EPSG:3577), 'sources': <xarray.DataArray (time: 1)>
array([ (Dataset <id=22a05adf-7559-4b10-89b0-e5b0dde8c213 type=ls5_nbar_albers location=/g/data/u46/users/gxr547/unicube/LS5_TM_NBAR/LS5_TM_NBAR_3577_15_-39_19930513231246500000.nc>,)], dtype=object)
Coordinates:
  * time     (time) datetime64[ns] 1993-05-13T23:12:46.500000}, 'tile_index': (15, -39, numpy.datetime64('1993-05-13T23:12:46.500000000'))}

```

Then it will ask to confirm the job should be submitted to PBS:
```
qsub -q normal -N fctest -P v10 -l ncpus=16,mem=31gb,walltime=1:00:00 -- /bin/bash "/g/data/v10/public/modules/agdc-fc/0.0.3/scripts/distributed.sh" --ppn 16 datacube-fc -v --load-tasks "/g/data2/v10/public/modules/agdc-fc/1.0.0/scripts/ls5_fc_albers_test_1993.bin" --executor distributed DSCHEDULER
RUN? [Y/n]:

```
