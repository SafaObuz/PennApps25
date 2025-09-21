# 4D2^3

** Create an editable 3D world just from a video **

Original Video | COLMAP Camera Trajectory (Poses) | Segmentation Maps (SAM)
:-------------:|:--------------------------------:|:----------------------:

![](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/video.gif) | ![](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/sam.gif) | ![](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/colmap.gif)


This is a simplified version of EgoLifter focused on stable Gaussian deletion functionality.

https://github.com/facebookresearch/egolifter

## Key Features

- **Object Deletion**: Remove unwanted Gaussians from 3D scenes
- **Movement Controls**: Move selected objects with precise controls
- **Stable Rendering**: Fixed CUDA/CPU device handling issues
- **Efficiency**: Modified original pipeline to run fully on a laptop

## Modified Files (Our Contributions)

- `edit_panel.py` - Main GUI with simplified interface
- `vanilla.py` - Fixed import and device handling for background tensor
- `gsplat.py` - Fixed import and device handling for K tensor
- `viewer_with_landing.py` - Landing page integration
- `setup_env.bash` - Environment configuration for lightweight SAM model
- `landing_page.html` - Web interface

As well as a handful of patches to the original EgoLifter codebase to enable training on a laptop w/ 4070

- Fixed import errors (relative → absolute imports)
- Fixed CUDA/CPU device handling
- Simplified GUI for better reliability

## Setup

1. Run `source setup_env.bash` to set up environment variables
2. Install dependencies from the original EgoLifter repository
3. Run `python viewer_with_landing.py` to start the viewer

## Usage

1. Use "Add Selection Panel" to create 3D selection grids and enable the pointcloud
2. Position grids to select unwanted Gaussians
3. Use "Delete Selected Objects" to remove them
4. Use movement controls to reposition objects if needed
