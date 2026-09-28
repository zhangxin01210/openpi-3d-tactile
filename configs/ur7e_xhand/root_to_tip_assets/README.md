# Root-to-Tip CAD Mesh Bundle

This directory contains a copy of `ur7e_xhand_verified.urdf` and the 76 mesh
files it references. The source is the local `3D_tactile` diagnostic project:

- `pointcloud_delivery/configs/ur7e_xhand_verified.urdf`
- `pointcloud_delivery/diagnostics/ur_description_source/meshes/ur5e/`
- `ur5_xhand/Flange_meshes/`
- `ur5_xhand/xhand_meshes/` and `ur5_xhand/xhand_obj/`

Only referenced files were copied. Their relative layout and file contents are
unchanged. The URDF matches `../ur7e_xhand_verified.urdf` byte for byte, so the
diagnostic renderer and the canonical FK use the same link geometry. This bundle
is for offline visualization; training and deployment do not load these meshes.
