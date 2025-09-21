# 4D2³

## **Create an editable 3D world just from a video**

Original Video | COLMAP Camera Trajectory (Poses) | Segmentation Maps (SAM)
:-------------:|:--------------------------------:|:----------------------:
![chair-video](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/video.gif) | ![chair-colmap](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/colmap.gif) | ![chair-sam](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/sam.gif)
![drexel-video](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_drexel/video.gif) | ![drexel-colmap](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_drexel/colmap.gif) | ![drexel-sam](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_drexel/sam.gif)
![penn-video](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_penn/video.gif) | ![penn-colmap](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_penn/colmap.gif) | ![penn-sam](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_penn/sam.gif)

### Valid view visualization (during training)

![training-outdoor](https://github.com/SafaObuz/PennApps25/blob/master/media/outdoor_penn/media_images_valid_view_frame_00736_1329_5aa010e87e8852a06a57.png)



![training-indoor](https://github.com/SafaObuz/PennApps25/blob/master/media/chair/media_images_valid_view_frame_00281_1063_014a6567d342deecc3e1.png)



This is a simplified version of EgoLifter focused on stable Gaussian editing (deletion, movement) functionality.

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
