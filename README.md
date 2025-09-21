# EgoLifter Scene Editor - Simplified Version

This is a simplified version of EgoLifter focused on stable Gaussian deletion functionality.

## Key Features

- **Grid-based Gaussian Selection**: Manual selection using 3D bounding boxes
- **Object Deletion**: Remove unwanted Gaussians from 3D scenes
- **Movement Controls**: Move selected objects with precise controls
- **Clean GUI**: Professional interface without emojis
- **Stable Rendering**: Fixed CUDA/CPU device handling issues

## Modified Files

- `edit_panel.py` - Main GUI with simplified interface
- `vanilla.py` - Fixed import and device handling for background tensor
- `gsplat.py` - Fixed import and device handling for K tensor
- `viewer_with_landing.py` - Landing page integration
- `setup_env.bash` - Environment configuration
- `landing_page.html` - Web interface

## Setup

1. Run `source setup_env.bash` to set up environment variables
2. Install dependencies from the original EgoLifter repository
3. Run `python viewer_with_landing.py` to start the viewer

## Usage

1. Use "Add Selection Panel" to create 3D selection grids
2. Position grids to select unwanted Gaussians
3. Use "Delete Selected Objects" to remove them
4. Use movement controls to reposition objects if needed

## Changes Made

- Removed complex uncertainty-based selection (was causing stability issues)
- Fixed import errors (relative → absolute imports)
- Fixed CUDA/CPU device handling
- Simplified GUI for better reliability
- Removed emojis for professional appearance

This version prioritizes stability and core functionality over advanced features.
