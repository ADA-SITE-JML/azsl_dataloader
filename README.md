## Data Loaders for the Azerbaijani Dactyl alphabet and Sign Language datasets

The repository contains the examples of data loaders written in Python, for the PyTorch framework, introduced as Jupyter Notebooks. This document describes the provided code in detail.

### Data loader for the dactyl alphabet

The dactyl signs consist of:
- short videos, that represent the letters signed as hand or wrist movements: 'D', 'Ü', 'Y', 'Ö', 'Z', 'C' and 'Ş' 
- images, that show the letter that does not require the motion of the hand, fingers or the wrist (all the remaining letter) 

### Data loader for the AzSL

The following diagram explains the flow of the word-based data loading process. It considers the following:
- based on the given video source (Cam1 or Cam2), feature type (i3d, squeezenet's last convolutional layer or just pixel data of frames) and video processing tool (OpenCV, VidGear or TorchVision) extracts the data from the mp4 files
- depending on the configuration, it may keep the frames with only hands, removing the idle parts of the video.

The output of the process is the encoded sentence and the corresponding video data. For debugging purposes, the name of the file is also added.

![AzSL_DataLoader1](https://github.com/ADA-SITE-JML/azsl_dataloader/assets/7048329/f32a23bf-59a1-474d-a653-d10885317504)

The data loader code starts with the initialization and config work, that defines the following:
- Possibility of running notebook on the host machine, not on Colab's environment. It is possble if the user has good computational resources and do not want to be interrupted after 8 hours or execution
- The source where the dataset is located: Global GoogleDrive, the user's Google Drive or any other accessible disk
- The target computation unit: CPU, GPU or TPU. The system will always give preference to GPU. For the TPU, it should be set explicitly setting _device = 'tpu'_
- The video processing library that the user prefers.  As an option, one of the OpenCV, VidGear or TorchVision could be selected. TorchVision is the default selection
- The camera source: a frontal or side camera could be used

The followingfigure graphically describes the flow:

![fig_data_loader](https://github.com/ADA-SITE-JML/azsl_dataloader/assets/7048329/28da6d1e-61c6-438f-9620-fa89156ad06d)

During the dataloading process the video frames are resized and converted to float tensors. A user may decide to have all the frames or only frames with the hands. If number of the frames with the hands is less than the _max_frames_, then the beginning and end are filled from the original video.

### References
The povided works are originally described in the following papers and expected to be cited as a reference:

1. J. Hasanov, N. Alishzade, A. Nazimzade, S. Dadashzade, T. Tahirov. _Development of a hybrid word recognition system and dataset for the Azerbaijani Sign Language dactyl alphabet_. Speech Communication, Volume 153, 2023, 102960, ISSN 0167-6393, https://doi.org/10.1016/j.specom.2023.102960.

2. N. Alishzade and J. Hasanov. _AzSLD: Azerbaijani Sign Language Dataset for Fingerspelling, Word, and Sentence Translation with Baseline Software._ The preprint is available at: https://arxiv.org/abs/2411.12865

The citing could be done through the GitHub's new "Cite this repository" function.
