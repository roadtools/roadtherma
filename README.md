# Analysis tool for thermal road data
This repository contains code for processing and analysing data from termal
cameras measured during paving operations. The tool has the following essential
features:

* Distinguish between road and non-road pixels based on temperature variations.
* Detection of clusters with high temperature differences on the identified road.
* Generation of several statistical measures of the temperature data and paving operation.

## Requirements & Installation
The roadtherma tool have been developed using python 3.8 but should *in theory*
also work with python 3.6+. The cli-tool is provided as a python package that
can be easily installed with `pip`. It is recommended to use pipenv to manage
dependencies & virtualenvironments. The installation guides below is based on
unix-based systems but should be almost identical to conda-environment on windows.

### Installing using `pipenv`
On a unix-based system, open a terminal and run the following in folder of your
choice

```
$ git clone https://github.com/roadtools/roadtherma.git
$ cd roadtherma
$ pipenv sync
$ pipenv run pip install ./
```

and now you can use the cli-tool as described in the next section, using the
virtual environment just created by `pipenv sync` above.

### Installing using `pip`
```
$ git clone https://github.com/roadtools/roadtherma.git
$ pip install -r roadtherma/requirements.txt
$ pip install  ./roadtherma
```
Note that you might have to use root for a system-wide installation.


## Using the cli-tool
To verify that roadtherma is installed, run:

```
$ roadtherma --help
```

and the following text should appear:

```
Usage: roadtherma [OPTIONS]

  Command line tool for analysing Pavement IR data. See
  https://github.com/roadtools/roadtherma for documentation on how to use
  this command line tool.

Options:
  --jobs_file TEXT  path of the job specification file (in YAML format).
                    [default: ./jobs.yaml]

  --help            Show this message and exit.

```

When invoked without arguments, the tool tries to open a "job file" 
at `./jobs.yaml` containing information on the data-files that should
be processed. An alternative location can be specified using the
`--jobs_file` flag. The jobfile is a YAML-file that lists one or more
"jobs". Each job consists of a path to a data-file together with
specifcations on how to parse the data-file, how to clean/process it, and
what output should be produced.

If more than one job is specified, each job inherits the
configuration settings from the previous job, unless explicitly set.
E.g. in,

```yaml
---
- job:
    title: Example 1 
    file_path: /path/to/data_file.txt
    reader: voegele_example
    write_csv: False # Overwriting the default value

- job:
    title: Example 2
    file_path: /path/to/second/data_file.txt

- job:
    title: Example 3
    file_path: /path/to/third/data_file.txt
```

both `Example 2` and `Example 3` will not produce csv-files,
even though the default behaviour is to do so, because it was
disabled in the configuration of the first job.

## Configuration parameters
An example files is given in `example.yaml` at the root of this repo,
where the first job sets all the configuration parameters to their
default values together with a short description of its use.
