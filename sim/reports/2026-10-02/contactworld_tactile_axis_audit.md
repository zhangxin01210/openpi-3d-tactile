# ContactWorld USB tactile axis audit (2026-10-02)

The USB task exposes one 10×14×3 force field, `tactile_force_field_right`.
The three channels are right-pad normal, local shear X, local shear Y. They
are sensor-local channels rather than robot-base X/Y/Z. The published data
and the current OpenPI datasets have no left-pad force field.

## Source-level findings

- `tacsl_env_insertion.py` defines both elastomer link names, but its active
  `tactile_shear_field_configs` contains only the right sensor (lines 82–104).
- `tacsl_task_usb.py` copies only `tactile_force_field_right` into observations
  (lines 416–423). The released Zarr therefore cannot supply left measurements.
- `tacsl_sensors.py` computes a penetration-depth normal term and a tangential
  relative-velocity term (lines 1456–1489). The coefficients are fixed at
  `kn=1`, `kt=0.1`, `mu=2`; a velocity-bias lookup is also subtracted in
  `query_collision` (lines 1118–1136). This is a synthetic pad/plug contact
  field, not a calibrated three-axis wrist force/torque sensor or the contact
  load exerted by the plug on the socket.
- Merely adding the left configuration to the list is unsafe: sensor geometry
  is currently cached by **indenter link ID** (lines 973–1000, 1324–1328),
  which is the same plug link for both fingers and would overwrite one pad's
  geometry. The tangential velocity correction also uses a hard-coded
  elastomer link ID 16 (line 1119). Both must be repaired and tested before
  a two-finger sensor can be claimed.

## Measured base-axis results

Right-pad channel order/signs were reconstructed from source as
`F_local_xyz=(-shear_x,-normal,shear_y)` and rotated through same-frame taxel
pose into the robot base; the local/base round trip passed. In the replayed
USB episode 12, pre-action frame 59 has 83 active taxels. Mean absolute
base-frame components are X `0.000438`, Y `0.000509`, Z `0.001284` in the
sensor's synthetic units. Thus Z can be the largest component during contact.
See [episode 12, frame 59](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_tactile_axis_audit_20261002/episode_012_frame_59.png).

In episode 1, frame 78 has a high socket axial contact load in the preceding
physics step (Z `13.24` in the simulator's contact-load units), while the
right-pad mean absolute base Z is only `0.0000305`, versus X `0.000461` and
Y `0.000570`. See [episode 1, frame 78](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_tactile_axis_audit_20261002/episode_001_frame_78.png).
These are different physical locations and different units, so their values
must not be equated. The example establishes that axial socket loading does
not guarantee a large right-pad Z signal.

Across the 32 paused replay captures, 308 frames had absolute post-action
socket Z load above 3. On the next observation frame, right-pad Z was the
largest mean absolute base component in 39.9% of those frames; in 14.6%, it
was below one tenth of the larger X/Y component. This is descriptive of these
captures, not a calibrated sensitivity estimate or proof of reliable jam
detection.

The absence of a visually obvious Z arrow in a 3D projection can reflect
view angle and a small Z component in that particular frame. The sensor's
right-pad normal points mostly along the grasp direction, while vertical
loading appears mainly in its shear channel. Because the shear model depends
on relative tangential velocity, a stationary jam need not produce a strong
or persistent vertical right-pad response.

## Decision for current pipeline

The pipeline is paused after 32/145 source demonstrations. No V2 full export
or V2 training has begun. Existing models use only the released right field;
they must be described as right-pad synthetic TacFF experiments. Before
claiming bilateral tactile sensing or reliable axial-force feedback, the
simulator must implement/test both pads or provide an independent wrist/socket
force sensor, respectively. Neither can be reconstructed from the released
right-only observations.
