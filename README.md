# MIMIC_race_ethnicity_annotations


The code in this repository is meant to help users read in and work with the annotated race and ethnicity sentences from MIMIC-III. No data is stored here and must be accessed through the PhsyioNet project page.

The `read_annotations_files_utils.py` provides utility functions to read in and validate data. The `read_data_files_example.py` script provides example code to use the utility functions and manipulate the data in the annotation files. While the script doesn't produce anything, it can be run using the following command: `python read_data_files_examples.py -ra dataSaves/raw_race_ethnicity_assignment_annotations/ -cd dataSaves/collected_data/` after race and ethnicity data hase been moved to the dataSaves folder.
