# scene/gsplat_model.py
import torch
from .gaussian_model import GaussianModel

class GsplatModel(GaussianModel):
    """
    Compatibility layer + group editing helpers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Lazy init: we don't know N yet (ckpt not loaded), so defer sizing.
        self.selected_mask = None  # not a registered buffer yet

    # ---- helpers to find/set XYZ tensor regardless of internal naming ----
    def _get_xyz_tensor(self) -> torch.Tensor:
        for name in ("xyz", "means3D", "_xyz"):
            t = getattr(self, name, None)
            if t is not None:
                return t
        # Some codepaths use a property 'get_xyz'
        if hasattr(self, "get_xyz"):
            return self.get_xyz
        raise RuntimeError("No xyz storage found on GaussianModel")

    def _set_xyz_tensor(self, new_xyz: torch.Tensor):
        for name in ("xyz", "means3D", "_xyz"):
            if hasattr(self, name):
                attr = getattr(self, name)
                if hasattr(attr, 'data'):
                    # It's a Parameter, set the data
                    attr.data = new_xyz
                else:
                    # It's a regular tensor, set directly
                    setattr(self, name, new_xyz)
                return
        # Fall back: some impls expose a setter; if not, this is an error
        raise RuntimeError("No xyz storage to write on GaussianModel")

    def _ensure_selected_mask(self):
        """Create/resize selection mask to match current number of Gaussians."""
        try:
            xyz = self._get_xyz_tensor()
        except RuntimeError:
            # Still not initialized (e.g., before ckpt load) — leave as None
            return
        N = xyz.shape[0]
        if self.selected_mask is None or self.selected_mask.numel() != N:
            # keep it simple: plain attribute is fine; no need to register_buffer
            device = xyz.device
            self.selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # ---- selection API used by the edit panel ----
    @torch.no_grad()
    def select(self, mask_or_indices):
        self._ensure_selected_mask()
        if self.selected_mask is None:
            # model still has no XYZ — ignore select until ckpt is loaded
            print("[GsplatModel] select: XYZ not ready yet; ignoring.")
            return

        m = self.selected_mask
        m.zero_()

        if isinstance(mask_or_indices, torch.Tensor) and mask_or_indices.dtype == torch.bool:
            if mask_or_indices.numel() != m.numel():
                print(f"[GsplatModel] select: mask size {mask_or_indices.numel()} != {m.numel()} (ignored)")
                return
            m |= mask_or_indices.to(m.device)
        else:
            idx = torch.as_tensor(mask_or_indices, device=m.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < m.numel())]
            if idx.numel() > 0:
                m[idx] = True

        print(f"[GsplatModel] select group size = {int(m.sum())}")

    @torch.no_grad()
    def translate_selected(self, dx=0.0, dy=0.0, dz=0.0):
        self._ensure_selected_mask()
        m = self.selected_mask
        if m is None or not m.any():
            print("[GsplatModel] translate_selected: empty or unavailable selection")
            return
        xyz = self._get_xyz_tensor()
        delta = torch.tensor([dx, dy, dz], device=xyz.device, dtype=xyz.dtype)
        new_xyz = xyz.clone()
        new_xyz[m] += delta
        self._set_xyz_tensor(new_xyz)
        print(f"[GsplatModel] translated {int(m.sum())} gaussians by ({dx:.3f}, {dy:.3f}, {dz:.3f})")
