# Super-Resolution-for-DRP-with-Experimental-Benchmarks
This repository contains supplementary materials for the paper "Resolution-Driven Errors in Digital Rock Physics and Their Mitigation Using Deep-Learning Super-Resolution with Experimental Benchmarks".  

We provide the full code, the training datasets, and the trained models for three examined sandstones: Berea, Parker, and Kentucky. 

# Results demonstration
The code in the corresponding folders enables ML training to enhance the resolution of segmented 3D micro-CT images for three sandstones:  
×16 for Berea sandstone (from 7 µm/voxel (a) to 0.44 µm/voxel (a′)),  
×16 for Parker sandstone (from 3 µm/voxel (b) to 0.19 µm/voxel (b′)),  
×32 for Kentucky sandstone (from 3 µm/voxel (c) to 94 nm/voxel (c′)).  

![Results demonstration](GH_images/Results_demonstration.jpg)  

# The full code  
For each rock, the training is provided separately. The full code for each sandstone can be found in the corresponding sample folder (Berea, Parker, or Kentucky), under the "code" folder  

# The training dataset  
The training dataset for each sandstone can be found in the corresponding sample folder, in the "data.zip" archive. Don't forget to unzip it before running the code!  

# The trained models  
The trained models for each sandstone can be found in the corresponding sample folder, in the "progress.zip" archive. Don't forget to unzip it before running the code also!

# Conda Environment
Before using this code, all required packages must be installed.    
For convenience, you may use the provided ```environment.yml``` file as follows:  
1. Create a new environment from the .yml file:
```
conda env create -f environment.yml
```
2. Activate the new environment once it’s created:
```
conda activate environment
```

# Training
The training can be launched from the ""code fodler with the following command:

```
python3 Architecture.py -d test --with_rotation -g_image_path Berea_7um_3.tif -d_image_path LR_M_-1.tif LR_M_0.tif LR_M_1.tif LR_M_2.tif Berea_CSLM_clay_gen.tif
```
where  

```-d``` The name of the directory to save the Generator in, under the 'progress' directory,     

```--with_rotation``` Use this option for data augmentaion (rotations and mirrors) of the High-Resolution input,      
   
```-g_image_path``` Relative path to the Low-Resolution 3D volume,    

```-d_image_path``` Relative path to the High-Resolution 2D slices for each Stage;

# Evaluation  
To use the pre-trained Generator for processing Low-Resolution image, launch the following command from the "code" folder 

```
python3 Evaluation.py -d test -volume_size_to_evaluate 256 256 256 -g_image_path test_7.tif
```
where  

```-d``` The name of the directory under the 'progress' directory where the pre-trained Generator parameters were saved,    
 
```-volume_size_to_evaluate``` The size of the Low-Resolution volume to be Super-Resolved,

```-g_image_path``` Relative path to the Low-Resolution image to Super-Resolve  





