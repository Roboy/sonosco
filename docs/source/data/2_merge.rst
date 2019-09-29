.. _merge:

Merging manifest files
========================

In order to merge multiple manifests into one, just specify a folder that contains all manifest
files to be merged and run the ``merge_manifest.py`` .
This will look for all .csv files and merge the content together in the specified output-file.

For example:
::
    python merge_manifest.py --merge-dir path/to/manifests_dir --output-path temp/manifests/merged_manifest.csv
