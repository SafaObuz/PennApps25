# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in nerfstudio
# the Regents of the University of California, Nerfstudio Team and contributors
# https://github.com/nerfstudio-project/nerfstudio/

import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re


class EditPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        # Initialize group-related attributes
        self.has_gaussian_groups = False
        self.selected_group_mask = None

        try:
            self._setup_point_cloud_folder()
            self._setup_gaussian_edit_folder()
            self._setup_save_gaussian_folder()
        except Exception as e:
            print(f"[EditPanel] Error initializing EditPanel: {e}")
            # Still try to set up basic functionality
            try:
                self._setup_point_cloud_folder()
                self._setup_save_gaussian_folder()
            except Exception as e2:
                print(f"[EditPanel] Critical error in EditPanel initialization: {e2}")

    def _setup_point_cloud_folder(self):
        server = self.server
        with self.server.add_gui_folder("Point Cloud"):
            self.show_point_cloud_checkbox = server.add_gui_checkbox(
                "Show Point Cloud",
                initial_value=False,
            )
            self.point_cloud_color = server.add_gui_vector3(
                "Point Color",
                min=(0, 0, 0),
                max=(255, 255, 255),
                step=1,
                initial_value=(0, 255, 255),
            )
            self.point_size = server.add_gui_number(
                "Point Size",
                min=0.,
                initial_value=0.01,
            )
            self.point_sparsify = server.add_gui_number(
                "Point Sparsify",
                min=1,
                initial_value=10,
            )

            self.pcd = None

            @self.show_point_cloud_checkbox.on_update
            @self.point_cloud_color.on_update
            @self.point_size.on_update
            @self.point_sparsify.on_update
            def _(event: viser.GuiEvent):
                with self.server.atomic():
                    self._update_pcd()

    def _resize_grid(self, idx):
        """Resize a grid based on the size handle"""
        try:
            if idx not in self.grids:
                return
                
            exist_grid = self.grids[idx][0]
            exist_grid.remove()
            
            # Get current position and rotation from transform controls
            grid_transform = self.grids[idx][1]
            grid_size = self.grids[idx][2].value
            
            self.grids[idx][0] = self.server.add_grid(
                "/grid/{}".format(idx),
                width=grid_size[0],
                height=grid_size[1],
                wxyz=grid_transform.wxyz,
                position=grid_transform.position,
            )
            self._update_scene()
            
        except Exception as e:
            print(f"[EditPanel] Error resizing grid {idx}: {e}")

    def _setup_gaussian_edit_folder(self):
        server = self.server

        with server.add_gui_folder("Edit"):
            # initialize a list to store panel(grid)'s information
            self.grids: dict[int, list[
                viser.MeshHandle,
                viser.TransformControlsHandle,
                viser.GuiInputHandle,
            ]] = {}
            self.grid_idx = 0

            # Instructions at the top
            server.add_gui_markdown("**How to Use:**")
            server.add_gui_markdown("1. Use 'Add Selection Panel' to manually select Gaussians")
            server.add_gui_markdown("2. Adjust selection parameters as needed")
            server.add_gui_markdown("3. Use movement controls to move selected objects")
            server.add_gui_markdown("4. Use 'Delete Selected' to remove unwanted Gaussians")
            
            # Traditional grid-based selection
            server.add_gui_markdown("**Gaussian Selection**")
            add_grid_button = server.add_gui_button("Add Selection Panel", icon=viser.Icon.PLUS)
            
            self.delete_gaussians_button = server.add_gui_button(
                "Delete Selected Objects",
                color="red",
                icon=viser.Icon.TRASH,
            )

            # Movement controls with better labeling
            server.add_gui_markdown("**Movement Controls**")
            self.dx = server.add_gui_number("Move X (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
            self.dy = server.add_gui_number("Move Y (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
            self.dz = server.add_gui_number("Move Z (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
            
            # Add some preset movement buttons
            server.add_gui_markdown("**Quick Movement**")
            move_up_btn = server.add_gui_button("Move Up (+Y)", icon=viser.Icon.ARROW_AUTOFIT_UP)
            move_down_btn = server.add_gui_button("Move Down (-Y)", icon=viser.Icon.ARROW_AUTOFIT_DOWN)
            move_left_btn = server.add_gui_button("Move Left (-X)", icon=viser.Icon.ARROW_AUTOFIT_LEFT)
            move_right_btn = server.add_gui_button("Move Right (+X)", icon=viser.Icon.ARROW_AUTOFIT_RIGHT)
            move_forward_btn = server.add_gui_button("Move Forward (+Z)", icon=viser.Icon.ARROW_AUTOFIT_RIGHT)
            move_back_btn = server.add_gui_button("Move Back (-Z)", icon=viser.Icon.ARROW_AUTOFIT_LEFT)
            
            self.move_btn = server.add_gui_button("Move Selected", icon=viser.Icon.ARROWS_MOVE)
            
            # Add callbacks for quick movement buttons
            @move_up_btn.on_click
            def _(_):
                # Reset other values, set only Y
                self.dx.value = 0.0
                self.dy.value = 0.1
                self.dz.value = 0.0
                self._move_selected_gaussians()
            
            @move_down_btn.on_click
            def _(_):
                self.dx.value = 0.0
                self.dy.value = -0.1
                self.dz.value = 0.0
                self._move_selected_gaussians()
            
            @move_left_btn.on_click
            def _(_):
                self.dx.value = -0.1
                self.dy.value = 0.0
                self.dz.value = 0.0
                self._move_selected_gaussians()
            
            @move_right_btn.on_click
            def _(_):
                self.dx.value = 0.1
                self.dy.value = 0.0
                self.dz.value = 0.0
                self._move_selected_gaussians()
            
            @move_forward_btn.on_click
            def _(_):
                self.dx.value = 0.0
                self.dy.value = 0.0
                self.dz.value = 0.1
                self._move_selected_gaussians()
            
            @move_back_btn.on_click
            def _(_):
                self.dx.value = 0.0
                self.dy.value = 0.0
                self.dz.value = -0.1
                self._move_selected_gaussians()

        # Add Gaussian Group Selection functionality
        self._setup_gaussian_group_folder()

        self.grid_folders = {}

        # create panel(grid)
        def new_grid(idx):
            with self.server.add_gui_folder("Grid {}".format(idx)) as folder:
                self.grid_folders[idx] = folder

                # TODO: add height
                grid_size = server.add_gui_vector2("Size", initial_value=(10., 10.), min=(0., 0.), step=0.01)

                grid = server.add_grid(
                    "/grid/{}".format(idx),
                    height=grid_size.value[0],
                    width=grid_size.value[1],
                )
                grid_transform = server.add_transform_controls(
                    "/grid_transform_control/{}".format(idx),
                    wxyz=grid.wxyz,
                    position=grid.position,
                )

                # resize panel on size value changed
                @grid_size.on_update
                def _(event: viser.GuiEvent):
                    with self.server.atomic():
                        self._resize_grid(idx)

                # handle panel deletion
                grid_delete_button = server.add_gui_button("Delete")

                @grid_delete_button.on_click
                def _(_):
                    with server.atomic():
                        try:
                            if idx in self.grids:
                                self.grids[idx][0].remove()
                                self.grids[idx][1].remove()
                                self.grids[idx][2].remove()
                                if idx in self.grid_folders:
                                    self.grid_folders[idx].remove()
                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            if idx in self.grids:
                                del self.grids[idx]
                            if idx in self.grid_folders:
                                del self.grid_folders[idx]

                    self._update_scene()

            # update the pose of panel(grid) when grid_transform updated
            @grid_transform.on_update
            def _(_):
                # Grids don't have wxyz, only position
                # Need to remove and recreate the grid with new position
                if idx in self.grids:
                    self.grids[idx][0].remove()
                    self.grids[idx][0] = server.add_grid(
                        "/grid/{}".format(idx),
                        width=self.grids[idx][2].value[0],
                        height=self.grids[idx][2].value[1],
                        wxyz=grid_transform.wxyz,
                        position=grid_transform.position,
                    )
                self._update_scene()

            self.grids[self.grid_idx] = [grid, grid_transform, grid_size]
            self._update_scene()

        # setup callbacks

        @add_grid_button.on_click
        def _(_):
            with server.atomic():
                new_grid(self.grid_idx)
                self.grid_idx += 1

        @self.delete_gaussians_button.on_click
        def _(_):
            with server.atomic():
                gaussian_to_be_deleted = self._get_selected_gaussians_mask()
                self.viewer.model.gaussians.delete_gaussians(gaussian_to_be_deleted)
                self._update_pcd()
            self.viewer.rerender_for_all_client()

        @self.move_btn.on_click
        def _(_):
            self._move_selected_gaussians()

    def _move_selected_gaussians(self):
        """Move the currently selected Gaussians"""
        try:
            with self.server.atomic():
                # Get group selection from your grid(s)
                mask = self._get_selected_gaussians_mask()
                if not mask.any():
                    print("[EditPanel] No Gaussians selected for movement")
                    return
                
                # Tell the model that this is the active selection
                if hasattr(self.viewer.model.gaussians, 'select'):
                    self.viewer.model.gaussians.select(mask)
                
                # Apply translation to the whole group
                if hasattr(self.viewer.model.gaussians, 'translate_selected'):
                    self.viewer.model.gaussians.translate_selected(
                        float(self.dx.value),
                        float(self.dy.value),
                        float(self.dz.value),
                    )
                else:
                    # Fallback: direct tensor manipulation using proper method
                    if hasattr(self.viewer.model.gaussians, '_set_xyz_tensor'):
                        xyz = self.viewer.model.gaussians.get_xyz
                        delta = torch.tensor([self.dx.value, self.dy.value, self.dz.value], 
                                           device=xyz.device, dtype=xyz.dtype)
                        new_xyz = xyz.clone()
                        new_xyz[mask] += delta
                        self.viewer.model.gaussians._set_xyz_tensor(new_xyz)
                    else:
                        print("[EditPanel] No suitable method found for moving Gaussians")
                        return
                
                print(f"[EditPanel] Moved {mask.sum().item()} Gaussians by ({self.dx.value:.3f}, {self.dy.value:.3f}, {self.dz.value:.3f})")
                
                # Refresh visuals
                self._update_pcd(mask)
            self.viewer.rerender_for_all_client()
        except Exception as e:
            print(f"[EditPanel] Error moving Gaussians: {e}")

    def _setup_gaussian_group_folder(self):
        """Setup UI for Gaussian group selection and manipulation"""
        server = self.server
        
        with server.add_gui_folder("Gaussian Groups"):
            # Check if we have a GaussianGrouping model
            try:
                self.has_gaussian_groups = (
                    hasattr(self.viewer.model, 'gaussians') and 
                    hasattr(self.viewer.model.gaussians, 'num_objects') and
                    self.viewer.model.gaussians.num_objects > 0
                )
            except Exception as e:
                print(f"[EditPanel] Error checking for Gaussian groups: {e}")
                self.has_gaussian_groups = False
            
            if self.has_gaussian_groups:
                try:
                    num_objects = self.viewer.model.gaussians.num_objects
                    self.group_options = [f"Group {i}" for i in range(num_objects)]
                    
                    # Group selection dropdown
                    self.group_dropdown = server.add_gui_dropdown(
                        "Select Group",
                        tuple(self.group_options),
                        initial_value=self.group_options[0] if self.group_options else None
                    )
                    
                    # Group manipulation buttons
                    self.select_group_btn = server.add_gui_button("Select Group", icon=viser.Icon.CHECK)
                    self.move_group_btn = server.add_gui_button("Move Group", icon=viser.Icon.ARROWS_MOVE)
                    self.delete_group_btn = server.add_gui_button("Delete Group", color="red", icon=viser.Icon.TRASH)
                    
                    # Group movement controls
                    self.group_dx = server.add_gui_number("Group ΔX (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
                    self.group_dy = server.add_gui_number("Group ΔY (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
                    self.group_dz = server.add_gui_number("Group ΔZ (meters)", min=-2.0, max=2.0, step=0.01, initial_value=0.0)
                    
                    # Group visualization
                    self.show_group_checkbox = server.add_gui_checkbox(
                        "Show Selected Group",
                        initial_value=False,
                        hint="Highlight the selected group in the point cloud"
                    )
                    
                    # Setup callbacks
                    self._setup_group_callbacks()
                except Exception as e:
                    print(f"[EditPanel] Error setting up Gaussian groups UI: {e}")
                    server.add_gui_markdown("**Error setting up Gaussian groups**")
                    server.add_gui_markdown(f"*Error: {str(e)}*")
            else:
                # For non-GaussianGrouping models, provide alternative object manipulation
                server.add_gui_markdown("**Object Manipulation Tools**")
                server.add_gui_markdown("*Use the grid-based selection tools below to manipulate objects*")
                
                # Add some helpful instructions
                server.add_gui_markdown("**How to manipulate objects:**")
                server.add_gui_markdown("1. Click 'Add Panel' to create a selection grid")
                server.add_gui_markdown("2. Position and resize the grid to select objects")
                server.add_gui_markdown("3. Use 'Move Selected' to translate selected objects")
                server.add_gui_markdown("4. Use 'Delete Gaussians' to remove selected objects")
                server.add_gui_markdown("5. Enable 'Show Point Cloud' to see selections")

    def _setup_group_callbacks(self):
        """Setup callbacks for Gaussian group operations"""
        
        @self.select_group_btn.on_click
        def _(_):
            if not self.has_gaussian_groups:
                return
            try:
                with self.server.atomic():
                    group_idx = self.group_dropdown.value
                    group_id = int(group_idx.split()[-1])  # Extract number from "Group X"
                    self._select_gaussian_group(group_id)
                    self._update_pcd()
                self.viewer.rerender_for_all_client()
            except Exception as e:
                print(f"[EditPanel] Error in select group callback: {e}")
        
        @self.move_group_btn.on_click
        def _(_):
            if not self.has_gaussian_groups:
                return
            try:
                with self.server.atomic():
                    group_idx = self.group_dropdown.value
                    group_id = int(group_idx.split()[-1])
                    self._move_gaussian_group(
                        group_id,
                        float(self.group_dx.value),
                        float(self.group_dy.value),
                        float(self.group_dz.value)
                    )
                    self._update_pcd()
                self.viewer.rerender_for_all_client()
            except Exception as e:
                print(f"[EditPanel] Error in move group callback: {e}")
        
        @self.delete_group_btn.on_click
        def _(_):
            if not self.has_gaussian_groups:
                return
            try:
                with self.server.atomic():
                    group_idx = self.group_dropdown.value
                    group_id = int(group_idx.split()[-1])
                    self._delete_gaussian_group(group_id)
                    self._update_pcd()
                self.viewer.rerender_for_all_client()
            except Exception as e:
                print(f"[EditPanel] Error in delete group callback: {e}")
        
        @self.show_group_checkbox.on_update
        def _(_):
            try:
                self._update_pcd()
            except Exception as e:
                print(f"[EditPanel] Error in show group checkbox callback: {e}")

    def _select_gaussian_group(self, group_id):
        """Select a Gaussian group by ID"""
        if not self.has_gaussian_groups:
            return
            
        try:
            # Get the object features for this group
            if hasattr(self.viewer.model, 'get_object_features'):
                # This is a GaussianGrouping model
                object_features = self.viewer.model.get_object_features()  # Shape: (N, num_objects)
                group_mask = object_features[:, group_id] > 0.5  # Threshold for group membership
                
                # Use the model's selection mechanism
                self.viewer.model.select(group_mask)
                
                # Store the mask for visualization
                self.selected_group_mask = group_mask
                    
                print(f"[EditPanel] Selected group {group_id} with {group_mask.sum().item()} Gaussians")
            else:
                print(f"[EditPanel] Model doesn't support group selection")
        except Exception as e:
            print(f"[EditPanel] Error selecting group {group_id}: {e}")

    def _move_gaussian_group(self, group_id, dx, dy, dz):
        """Move a Gaussian group by the specified translation"""
        if not self.has_gaussian_groups:
            return
            
        try:
            # First select the group
            self._select_gaussian_group(group_id)
            
            # Then move the selected Gaussians
            if hasattr(self.viewer.model, 'translate_selected'):
                self.viewer.model.translate_selected(dx, dy, dz)
                print(f"[EditPanel] Moved group {group_id} by ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            else:
                print(f"[EditPanel] Model doesn't support group movement")
        except Exception as e:
            print(f"[EditPanel] Error moving group {group_id}: {e}")

    def _delete_gaussian_group(self, group_id):
        """Delete a Gaussian group"""
        if not self.has_gaussian_groups:
            return
            
        try:
            # First select the group
            self._select_gaussian_group(group_id)
            
            # Then delete the selected Gaussians
            if hasattr(self.viewer.model, 'delete_gaussians'):
                # Get the current selection mask
                if hasattr(self.viewer.model, 'selected_mask'):
                    mask = self.viewer.model.selected_mask
                    self.viewer.model.delete_gaussians(mask)
                    print(f"[EditPanel] Deleted group {group_id}")
                else:
                    print(f"[EditPanel] No selection mask available for deletion")
            else:
                print(f"[EditPanel] Model doesn't support group deletion")
        except Exception as e:
            print(f"[EditPanel] Error deleting group {group_id}: {e}")

    def _setup_save_gaussian_folder(self):
        with self.server.add_gui_folder("Save"):
            name_text = self.server.add_gui_text(
                "Name",
                initial_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_button = self.server.add_gui_button("Save")

            @save_button.on_click
            def _(event: viser.GuiEvent):
                # skip if not triggered by client
                if event.client is None:
                    return
                try:
                    save_button.disabled = True

                    with self.server.atomic():
                        try:
                            # check whether is a valid name
                            name = name_text.value
                            match = re.search("^[a-zA-Z0-9_\-]+$", name)
                            if match:
                                # save ply
                                ply_save_path = os.path.join("edited", "{}.ply".format(name))
                                self.viewer.model.gaussians.to_ply_structure().save_to_ply(ply_save_path)
                                message_text = "Saved to {}".format(ply_save_path)

                                # save as a checkpoint if viewer started from a checkpoint
                                if self.viewer.checkpoint is not None:
                                    checkpoint_save_path = os.path.join("edited", "{}.ckpt".format(name))
                                    checkpoint = self.viewer.checkpoint
                                    # update state dict of the checkpoint
                                    state_dict_value = self.viewer.model.gaussians.to_parameter_structure()
                                    for name_in_dict, name_in_dataclass in [
                                        ("xyz", "xyz"),
                                        ("features_dc", "features_dc"),
                                        ("features_rest", "features_extra"),
                                        ("scaling", "scales"),
                                        ("rotation", "rotations"),
                                        ("opacity", "opacities"),
                                    ]:
                                        dict_key = "gaussian_model._{}".format(name_in_dict)
                                        assert dict_key in checkpoint["state_dict"]
                                        checkpoint["state_dict"][dict_key] = getattr(state_dict_value, name_in_dataclass)
                                    # save
                                    torch.save(checkpoint, checkpoint_save_path)
                                    message_text += " & {}".format(checkpoint_save_path)
                            else:
                                message_text = "Invalid name"
                        except:
                            traceback.print_exc()

                    # show message
                    with event.client.add_gui_modal("Message") as modal:
                        event.client.add_gui_markdown(message_text)
                        close_button = event.client.add_gui_button("Close")

                        @close_button.on_click
                        def _(_) -> None:
                            modal.close()

                finally:
                    save_button.disabled = False

    def _get_selected_gaussians_mask(self):
        xyz = self.viewer.model.gaussians.get_xyz

        # if no grid exists, do not delete any gaussians
        if len(self.grids) == 0:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.bool)

        # initialize mask with True
        is_gaussian_selected = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
        for i in self.grids:
            # get the pose of grid, and build world-to-grid transform matrix
            grid = self.grids[i][0]
            se3 = torch.linalg.inv(torch.tensor(vtf.SE3.from_rotation_and_translation(
                vtf.SO3(grid.wxyz),
                grid.position,
            ).as_matrix()).to(xyz))
            # transform xyz from world to grid
            new_xyz = torch.matmul(xyz, se3[:3, :3].T) + se3[:3, 3]
            # find the gaussians to be deleted based on the new_xyz
            grid_size = self.grids[i][2].value
            x_mask = torch.abs(new_xyz[:, 0]) < grid_size[0] / 2
            y_mask = torch.abs(new_xyz[:, 1]) < grid_size[1] / 2
            #z_mask = new_xyz[:, 2] > 0
            # update mask
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, x_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, y_mask)
            #is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, z_mask)

        return is_gaussian_selected

    def _update_pcd(self, mask=None):
        """Update point cloud visualization"""
        try:
            if not self.show_point_cloud_checkbox.value:
                if self.pcd is not None:
                    self.pcd.remove()
                    self.pcd = None
                return

            xyz = self.viewer.model.gaussians.get_xyz
            colors = self.viewer.model.gaussians.get_colors

            # Apply sparsification
            sparsify = int(self.point_sparsify.value)
            if sparsify > 1:
                indices = torch.arange(0, xyz.shape[0], sparsify, device=xyz.device)
                xyz = xyz[indices]
                colors = colors[indices]

            # Convert colors to RGB if needed
            if colors.shape[1] == 3:
                rgb_colors = colors
            else:
                # Convert from spherical harmonics or other format
                rgb_colors = torch.clamp(colors[:, :3], 0, 1)

            # Apply custom color if specified
            if hasattr(self, 'point_cloud_color'):
                custom_color = np.array([
                    self.point_cloud_color.value[0] / 255.0,
                    self.point_cloud_color.value[1] / 255.0,
                    self.point_cloud_color.value[2] / 255.0,
                ])
                rgb_colors = torch.full_like(rgb_colors, custom_color)

            # Highlight selected points if mask provided
            if mask is not None and mask.any():
                # Make selected points more visible
                rgb_colors[mask] = torch.tensor([1.0, 0.0, 0.0], device=rgb_colors.device)

            # Convert to numpy for viser
            xyz_np = xyz.detach().cpu().numpy()
            colors_np = rgb_colors.detach().cpu().numpy()

            # Update or create point cloud
            if self.pcd is not None:
                self.pcd.remove()
            
            self.pcd = self.server.add_point_cloud(
                "/point_cloud",
                points=xyz_np,
                colors=colors_np,
                point_size=self.point_size.value,
            )

        except Exception as e:
            print(f"[EditPanel] Error updating point cloud: {e}")

    def _update_scene(self):
        """Update the scene visualization"""
        try:
            # Update point cloud if it's being shown
            if hasattr(self, 'show_point_cloud_checkbox') and self.show_point_cloud_checkbox.value:
                self._update_pcd()
        except Exception as e:
            print(f"[EditPanel] Error updating scene: {e}")