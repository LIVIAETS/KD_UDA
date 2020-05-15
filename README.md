# KD-UDA
This is the repository for Joint Progressive Knowledge Distillation andUnsupervised Domain Adaptation.

# Requirements
- pytorch=>1.1.0
- visdom
- https://github.com/luizgh/visdom_logger

# Datasets:
- Office31: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code
- ImageClef-DA: https://www.imageclef.org/2014/adaptation

# How to run:
One you have installed all the requirements and download the dataset. Please take a look at the file kd_da_alt.py.
All the hyper-parameters can be found in the main function(). By default we assume the dataset to be at, for example webcam: "~/datasets/webcam/images"

Once you have configured all paths, you can use:
``python kd_da_alt.py`` to launch a training

# Acknowledgements
We used some of the code from:
- https://github.com/thuml/CDAN: to load ImageClef dataset
- https://github.com/jindongwang/transferlearning/tree/master/code/deep/TCP: to load Office31 dataset and MMD loss
