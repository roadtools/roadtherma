import numpy as np

# Temperature threshold for the trimming step
trim_threshold = 80

# Percentage of data that should be above trim_threshold in order for that outer longitudinal line to be removed
percentage_above = 0.2

# Temperature threshold for the road length estimation step
roadlength_threshold = 80

# Tolerance on the temperature difference during temperature gradient detection
gradient_tolerance = 10

# Array/list of tolerance-values to be used in generating plots of percentage-of-road-that-a-high-temperature-gradient vs gradient-tolerance
tolerances = np.arange(5, 20, 1)

# Data files that should be processed.
# Format for specifying a data-file: (title, filepath, reader) where reader is the appropriate parser for that file
data_files = [
        ('plot title', 'string containing data file path', 'string with reader-type'),
        ('another plot title', 'another string containing data file path', 'another string with reader-type'),
        ('yet another plot title', 'yet another string containing data file path', 'yet another string with reader-type'),
        ]
