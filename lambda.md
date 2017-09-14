
# The plan for running jobs from AWS Lambda

Required arguments passed Lambda:

* **Year** _or range or years_
* **Project** Such as `v10`/`u46`
* **FC config** _defines which product to process_
* **Environment** _which module/environment to run in_
* **processing-dir**: Output directory for logs and `task-files`. Will be of the form `/g/data/v10/work/fc/<prod_name>/<submission-time>`


## Lambda executes this script on `raijin`
The result of which is submitting a PBS job to generate tasks and then
queue the execution of them.
```
module use /g/data/v10/public/modules/modulefiles
module load agdc-py3-prod
module load agdc-fc
cd ${PROCESSING_DIR}
datacube-fc qsub-task-generation —project ${PROJECT} —app-config ${APP_CONFIG} —year ${YEAR} —save-tasks <task-file-name>
```

The above generates a PBS job which calls the following:

## PBS Job for Task Generation of Queueing of Processing

1. **Generates tasks** into `<path-to-taskfile>`
2. **Calculates required job size** to process these tasks (nodes/ram/time)
3. **Queues** a PBS job of the right size to process the task

The job doesn't exist as a script, but if it did it would look like:
```
module use /g/data/v10/public/modules/modulefiles
module load agdc-py3-prod
module load agdc-fc

datacube-fc initiate-batch-processsing <app-config> <year> <processing-dir>
```



## PBS Job for Processing Fractional Cover tasks

Again, doesn't exist as a script on disk, but if it did, it would look like:

```
module use /g/data/v10/public/modules/modulefiles
module load agdc-py3-prod
module load agdc-fc

datacube-fc process-tasks <path-to-taskfile>
```



## Extra command for running manually

`datacube-fc run-fc <app-config>`