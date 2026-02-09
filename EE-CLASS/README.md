# echocv

### Installation

Our echo computer vision pipeline has a several dependencies including:

1. opencv
2. tensorflow-gpu
3. trackpy
4. PIMS
5. numpy/scipy/pandas/scikit-learn
6. gdcm (Grassroots DICOM library)

We recommend using Anaconda, which has many of these installed, and allows ready installation of the remaining packages through conda or pip. This project is written in python2. Compatible library versions can be found in `requirements.txt`. 

##### Creating an virtual environment using Anaconda

1. Create a virtual conda environment: `conda create -n echocv python=2.7.11 anaconda`
2. Activate that environment: `source activate echocv`
3. Install the libraries into the echocv environment: `pip install -r requirements.txt`

### Usage

Our workflow takes as input a folder of DICOM format videos and performs the following

1. View classification:  using predict_viewclass_v2.py
2. Segmentation: using segment_a4c_a2c_a3c_plax_psax.py
3. Common 2D measurements including mass, volumes and ejection fraction using analyze_segmentations_unet_a4c_a2c_psax_plax_v2.py
4. Global longitudinal strain: using strain_strainrate_unet_pool_v2.py

Models are currently stored separately on Dropbox:  

https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0

License details are provided in license.txt - and basically allow academic and non-profit use.
