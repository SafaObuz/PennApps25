# Fixed indentation for lines 35-46
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
