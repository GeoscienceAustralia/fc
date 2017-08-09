Fractional Cover (fc)
=====================

|Build Status| |Coverage Status|

Fractional Cover measures the photosynthetic, non-photosynthetic and
bare earth components of a Landsat image.

Installation
------------

To install the module on raijin:

Update the file metadata config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to pull the latest file metadata (netCDF global attributes) that are to be written to the output files from the CMI system.

If you are on raijin or VDI::

    module use /g/data/v10/public/modules/modulefiles
    module use /g/data/v10/private/modules/modulefiles
    module load dea-prod/********  # (Use appropriate version)
    
Otherwise install the ``digitalearthau`` package from https://github.com/GeoscienceAustralia/digitalearthau
      
Then harvest the metadata::

    dea-harvest 119

Ensure that the global attributes from the harvest match the global attributes
in the config files, and update appropriately.

Download from GitHub
~~~~~~~~~~~~~~~~~~~~

Checkout the tagged branch you wish to install to temp directory::

    git clone git@github.com:GeoscienceAustralia/fc.git
    cd fc
    git checkout tags/1.0.2
    git describe --tags --always

The tagged version should be printed.

Then to install::

    $ module use /g/data/v10/public/modules/modulefiles/
    $ ./package-module.sh 

You will be promted to check the package location and version. If it is
correct, type **``y``** and press enter

::

    # Packaging agdc-fc v 1.0.0 to /g/data/v10/public/modules/agdc-fc/1.0.0 #
    Continue? 

Setup on VDI
~~~~~~~~~~~~

The first time you try to use raijin PBS commands from VDI, you will need
to run::

    $ remote-hpc-cmd init

See http://vdi.nci.org.au/help#heading=h.u1kl1j7vdt16 for more details.

You will also need to setup datacube to work from VDI and rajin.

::

    $ ssh raijin "cat .pgpass" >> ~/.pgpass
    $ chmod 0600 ~/.pgpass

See http://geoscienceaustralia.github.io/digitalearthau/connect/nci_basics.html for
full details.

Running
-------

The Fractional Cover application works in 2 parts:

    #. Creating the task list
    #. Check for unexpected existing files - these were most likely created during an run that did not successfully finish.
    #. Submit the job to raijin.

To run fractional cover::

    $ module use /g/data/v10/public/modules/modulefiles/
    $ module load agdc-fc
    $ datacube-fc-launcher list

This will list the availiable app configs::

    ls5_fc_albers.yaml
    ls7_fc_albers.yaml
    ls8_fc_albers.yaml

To submit the job to `raijin`, the launcher has a the ``qsub`` command:

Usage: ``datacube-fc-launcher qsub [OPTIONS] APP_CONFIG YEAR``

Options:

* ``-q, --queue normal``            The queue to use, either ``normal`` or ``express``
* ``-P, --project v10``             The project to use
* ``-n, --nodes INTEGER RANGE``     Number of *nodes* to request  [required]
* ``-t, --walltime 4``              Number of *hours* to request
* ``--name TEXT``                   Job name to use
* ``--config PATH``                 Datacube config file (be default uses the currently loaded AGDC module)
* ``--env PATH``                    Node environment setup script (by default uses the installed production environment)
* ``--help``                        Show help message.

Change your working directory to a location that can hold the task file, 
and run the launcher specifying the app config, year (``1993`` or a range ``1993-1996``), and PBS properties:
::

    $ cd /g/data/v10/tmp
    $ datacube-fc-launcher qsub ls5_fc_albers.yaml 1993-1996 -q normal -P v10 -n 25 -t 1
    
We have found for best throughput *25 nodes* can produce about 11.5 tiles per minute per node, with a CPU efficiency of about 96%.

It will check to make sure it can access the database::

    Version: 1.1.9
    Read configurations files from: ['/g/data/v10/public/modules/agdc-py2-prod/1.1.9/datacube.conf']
    Host: 130.56.244.227:6432
    Database: datacube
    User: adh547


    Attempting connect
    Success.

Then it will create the task file in the current working directory, and create the output product
definition in the database (if it doesn't already exist)::

    datacube-fc -v --app-config "/g/data/v10/public/modules/agdc-fc/1.0.0/config/ls5_fc_albers.yaml" --year 1993-1996 --save-tasks "/g/data/v10/tmp/ls5_fc_albers_test_1993-1996.bin"
    RUN? [Y/n]:

    2016-07-13 18:38:56,308 INFO Created DatasetType ls5_fc_albers
    2016-07-13 18:39:01,997 INFO 291 tasks discovered
    2016-07-13 18:39:01,998 INFO 291 tasks discovered
    2016-07-13 18:39:02,127 INFO Saved config and tasks to /g/data/v10/tmp/ls5_fc_albers_test_1993-1996.bin

It can then list every output file to be created and check that it does not yet exist::

    datacube-fc -v --load-tasks "/g/data/v10/tmp/ls5_fc_albers_1993-1996.bin" --dry-run
    RUN? [y/N]:

    Starting Fractional Cover processing...
    Files to be created:
    /g/data/fk4/datacube/002/LS5_TM_FC/15_-39/LS5_TM_FC_3577_15_-39_19930513231246500000.nc
    /g/data/fk4/datacube/002/LS5_TM_FC/15_-40/LS5_TM_FC_3577_15_-40_19930513231246500000.nc
    ...
    144 tasks files to be created (144 valid files, 0 existing paths)
    
If any output files already exist, you will be asked if they should be deleted.

Then it will ask to confirm the job should be submitted to PBS::

    qsub -q normal -N fctest -P v10 -l ncpus=16,mem=31gb,walltime=1:00:00 -- /bin/bash "/g/data/v10/public/modules/agdc-fc/1.0.0/scripts/distributed.sh" --ppn 16 datacube-fc -v --load-tasks "/g/data/v10/tmp/ls5_fc_albers_1993-1996.bin" --executor distributed DSCHEDULER
    RUN? [Y/n]:

It should then return a job id, such as ``7517348.r-man2``

If you say `no` to the last step, the task file you created can be submitted to qsub later by calling::

    datacube-fc-launcher qsub -q normal -P v10 -n 1 --taskfile "/g/data/v10/tmp/ls5_fc_albers_1991-1992.bin" ls5_fc_albers.yaml


Tracking progress
-----------------

::

    $ qstat -u $USER

    $ qcat 7517348.r-man2 | head

    $ qcat 7517348.r-man2 | tail

    $ qps 7517348.r-man2

(TODO: Add instructions to connect to ``distributed`` web interface...)


File locations
--------------

The config file (eg. ls5_fc_albers.yaml) specifies the app settings, and is found in the module.

You will need to check the folder of the latest ``agdc-fc`` module::

    ls /g/data/v10/public/modules/agdc-fc/

To view the app config file, replace ``1.0.0`` with the latest version from above. 
::

    head /g/data/v10/public/modules/agdc-fc/1.0.0/config/ls5_fc_albers_test.yaml
    
The config file lists the output `location` and file_path_template``, as shown in this snippet::

    source_type: ls5_nbar_albers
    output_type: ls5_fc_albers
    version: 1.0.0
    
    description: Landsat 5 Fractional Cover 25 metre, 100km tile, Australian Albers Equal Area projection (EPSG:3577)
    product_type: fractional_cover
    
    location: '/g/data/fk4/datacube/002/'
    file_path_template: 'LS5_TM_FC/{tile_index[0]}_{tile_index[1]}/LS5_TM_FC_3577_{tile_index[0]}_{tile_index[1]}_{start_time}.nc'

So here the output files are saved to ``/g/data/fk4/datacube/002/LS5_TM_FC/<tile_index>/*.nc``


.. |Build Status| image:: https://travis-ci.org/GeoscienceAustralia/fc.svg?branch=master
    :target: https://travis-ci.org/GeoscienceAustralia/fc
    
.. |Coverage Status| image:: https://coveralls.io/repos/github/GeoscienceAustralia/fc/badge.svg?branch=master
    :target: https://coveralls.io/github/GeoscienceAustralia/fc?branch=master
