Fractional Cover (fc)
=====================

|Build Status| |Coverage Status|

Fractional Cover measures the photosynthetic, non-photosynthetic and
bare earth components of a Landsat image.

Fractional cover is available as a part of Digital Earth Australia environment modules on the NCI.
These can be used after logging into the NCI and running:

    module load dea

Setup on VDI
~~~~~~~~~~~~

The first time you try to use raijin PBS commands from VDI, you will need
to run::

    $ remote-hpc-cmd init

See http://vdi.nci.org.au/help#heading=h.u1kl1j7vdt16 for more details.

You will also need to setup datacube to work from VDI and raijin.

::

    $ ssh raijin "cat .pgpass" >> ~/.pgpass
    $ chmod 0600 ~/.pgpass

See http://geoscienceaustralia.github.io/digitalearthau/connect/nci_basics.html for
full details.

Running
-------

The Fractional Cover application works in 2 parts:

    #. Creating the task list
    #. Check for unexpected existing files - these were most likely created during an run that did not successfully
       finish.
    #. Submit the job to raijin.

To run fractional cover::

    $ module use /g/data/v10/public/modules/modulefiles/
    $ module load dea

This will list the available app configs::
    $ datacube-fc list
    ls5_fc_albers.yaml
    ls7_fc_albers.yaml
    ls8_fc_albers.yaml

To submit the job to `raijin`, the datacube-fc app has a the ``datacube-fc submit`` command:
This command kick off two stage PBS job
    Stage 1 (Generate task file):
        The task-app machinery loads a config file, from a path specified on the
        command line, into a dict.

        If dry is enabled, a dummy DatasetType is created for tasks generation without indexing
        the product in the database.
        If dry run is disabled, generate tasks into file and queue PBS job to process them.

    Stage 2 (Run):
        During normal run, following are performed:
           1) Tasks (loadables (nbart,ps,dsm) + output targets) shall be yielded for dispatch to workers.
           2) Load data
           3) Run FC algorithm
           4) Attach metadata
           5) Write output files and
           6) Finally index the newly created FC output files

        If dry run is enabled, application only prepares a list of output files to be created and does not
        record anything in the database.

Usage: ``datacube-fc submit [OPTIONS]``

Options:

* ``-q, --queue [normal|express]``                 The queue to use, either ``normal`` or ``express``
* ``-P, --project TEXT``                           The project to use
* ``--year TEXT``                                  Limit the process to a particular year
* ``--no-qsub``                                    Skip submitting job
* ``--tag TEXT``                                   Unique id for the job
* ``--app-config PATH``                            Fractional Cover app configuration file
* ``--dry-run``                                    Do not record anything into the database
* ``-m, --email-options [a|b|e|n|ae|ab|be|abe]``   Send Email when execution is,
                                                     [a = aborted | b = begins | e = ends | n = do not send email]
* ``-M, --email-id TEXT``                          Email Recipient List
* ``-C, --config, --config_file TEXT``             Datacube config file
* ``--local``                                      Experimental option. Execute fractional cover in the local environment. 
* ``-v, --verbose``                                Use multiple times for more verbosity
* ``--help``                                       Show help message.

Tracking progress
-----------------

::

    $ qstat -u $USER

    $ qcat 7517348.r-man2 | head

    $ qcat 7517348.r-man2 | tail

    $ qps 7517348.r-man2

File locations
--------------

The config file (eg. ls5_fc_albers.yaml) specifies the app settings, and is found in the module.

You will need to check the folder of the latest ``dea`` module::

    ls /g/data/v10/public/modules/dea/<YYYYMMDD>/lib/python3.6/site-packages/fc/config

The config file lists the output `location` and file_path_template``, as shown in this snippet::

    source_type: ls5_nbar_albers
    output_type: ls5_fc_albers

    description: Landsat 5 Fractional Cover 25 metre, 100km tile, Australian Albers Equal Area projection (EPSG:3577)
    product_type: fractional_cover
    
    location: '/g/data/fk4/datacube/002/'
    file_path_template: 'LS5_TM_FC/{tile_index[0]}_{tile_index[1]}/LS5_TM_FC_3577_{tile_index[0]}_{tile_index[1]}_{start_time}_v{version}.nc'

So here the output files are saved to ``/g/data/fk4/datacube/002/FC/LS5_TM_FC/<tile_index>/*.nc``

License
-------
This repository is licensed under the Apache License 2.0. See the `LICENSE file <LICENSE>`_ in this repository for details.


Contacts
--------
Geoscience Australia developers:

**Joshua Sixsmith**
joshua.sixsmith@ga.gov.au

**Jeremy Hooke**
jeremy.hooke@ga.gov.au

**Damien Ayers**
damien.ayers@ga.gov.au

**Duncan Gray**
duncan.gray@ga.gov.au

Algorithm developer:

**Peter Scarth**
peter.scarth@qld.gov.au


.. |Build Status| image:: https://travis-ci.org/GeoscienceAustralia/fc.svg?branch=master
    :target: https://travis-ci.org/GeoscienceAustralia/fc
    
.. |Coverage Status| image:: https://coveralls.io/repos/github/GeoscienceAustralia/fc/badge.svg?branch=master
    :target: https://coveralls.io/github/GeoscienceAustralia/fc?branch=master
