This set of scripts is designed to train and test classifiers for images of phytoplankton from the Imaging FlowCytobot (IFCB). It is written by Karin Garefelt in the Environmental Genomics group at Science For Life Laboratory/KTH, with inspiration from Kaisa Kraft et al. Front. Mar. Sci. (2022).

The scripts are designed to be run in this order:

- train.py to train the network
- determine_thresholds.py to determine thresholds to use for classification (otherwise leave images unclasssified)
- predict.py to predict classes for new images


test_on_test_data.py is for evaluating the classifiers performance. 
count_files.py is for counting files to determine the size of training set
extract_deep_features.py is for extracting deep features, fitting clustering models, all very experimental.
