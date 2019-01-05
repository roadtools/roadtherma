## Analysis tool for thermal road data
This repository contains code for processing and analysing data from termal
cameras measured during paving operations. The tool has the following essential
features:

* Distinguish between road and non-road pixels based on temperature variations.
* Detection of clusters with high temperature differences on the identified road.
* Generation of several statistical measures of the temperature data and paving operation.

*note that this is code is in a very early stage of development*

### Installation
On unix-based systems open a terminal and run the following

```
$ git clone https://github.com/roadtools/roadtherma.git
$ pip3 install roadtherma
```

and thats it! You sould now be able to use the CLI tool described below.

### CLI Tool
The file `pave_analyser.py` is the executable CLI. It can be terminal or
within IPython or a similar environment. Below is shown how to print the
help screen containing usage instructions.


From a terminal:
```
$ roadtherma --help
```

From IPython:
```
In [21]: !roadtherma --help
```

This should produce the following text:

```
Usage: roadtherma [OPTIONS]

  Command line tool for analysing Pavement IR data. It assumes that a file
  './data_files.py' (located where this script is executed) exists and
  contains a list of tuples named 'data_files' as follows:

      data_files = [

          (title, filepath, reader, width),

          ...,

          ]

  where 'title' is a string used as title in plots, 'filepath' contains the
  path of the particular data file, 'reader' is name of apropriate parser
  for that file and can have values "TF", "voegele_taulov",
  "voegele_example" or "voegele_M119". 'width' is a float containing the
  resolution in meters of the pixels in the transversal direction (the
  horizontal resolution is derived from the chainage data). Options to
  configure the data processing is specified below.

Options:
  --plots / --no-plots            Whether or not to create plots.  [default:
                                  True]
  --cache / --no-cache            Whether or not to use caching. If this is
                                  enabled and no caching files is found in
                                  "./.cache" the data will be processed from
                                  scratch and cached afterwards. The directory
                                  "./cache" must exist for caching to work.
                                  [default: True]
  --savefig / --no-savefig        Wheter or not to save the generated plots as
                                  png-files.  [default: False]
  --trim_threshold FLOAT          Temperature threshold for the data trimming
                                  step.  [default: 80.0]
  --percentage_above FLOAT        Percentage of data that should be above
                                  trim_threshold in order for that outer
                                  longitudinal line to be removed.  [default:
                                  0.2]
  --roadwidth_threshold FLOAT     Temperature threshold for the road width
                                  estimation step.  [default: 80.0]
  --adjust_npixel INTEGER         Additional number of pixels to cut off edges
                                  during road width estimation.  [default: 2]
  --gradient_tolerance FLOAT      Tolerance on the temperature difference
                                  during temperature gradient detection.
                                  [default: 10.0]
  --cluster_threshold_npixels INTEGER
                                  Minimum amount of pixels that should be in a
                                  cluster. Clusters below this value will be
                                  discarded.  [default: 0]
  --cluster_threshold_sqm FLOAT   Minimum size of a cluster in square meters.
                                  Clusters below this value will be discarded.
                                  [default: 0.0]
  --tolerance_range <INTEGER INTEGER INTEGER>...
                                  Range of tolerance values (e.g. "--
                                  tolerance_range <start> <end> <step size>")
                                  to use when plotting percentage of road that
                                  is comprised of high gradients vs gradient
                                  tolerance.  [default: 5, 20, 1]
  --help                          Show this message and exit.
 ```
