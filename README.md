## DL Final Project

## Progress

* librispeech.py works
   * Currently loads all data in a folder. Consider adding an option to load only some data
* onehot.py should work
* CNN is probably correct
* Transformer seems to work (untested)
* Model structure is almost completely wrong
* mask.py has some okay parts but needs to be reformatted and tested
* Losses are nonexistent
* train.py should be ignored

## Data

LibriSpeech: https://www.openslr.org/12

I recommend downloading the dev-clean and test-clean data sets, since those are smaller in size.

If you extract the data under `data/`, the .gitignore will prevent it from being uploaded.

## Prerequisites

Install Tensorflow and [soundfile](https://pypi.org/project/soundfile/).
