# ContactWorld USB round 2: input contract and training gate

## Scope

This round is a targeted follow-up to the USB V3 13-model matrix.  It keeps
the reviewed bilateral USB data, two RGB images, 18-D proprioception, 16-step
action horizon, Pi0 LoRA base weights, seed 42, batch size 8, 20,000 updates,
cosine schedule, and suffix conditioning fixed.  The six models below isolate
three questions around the first-round leaders 07 and 10.

| ID | Model | Changed input | Tokens sent to the action suffix | Question |
|---|---|---|---|---|
| 14 | `base_pooled_no_pc` | 07 without the 4096-point cloud | 6 tactile | Is 07's gain retained without point cloud? |
| 15 | `base_map_matched_no_pc` | 10 without the 4096-point cloud | 6 tactile | Is 10's gain retained without point cloud? |
| 16 | `contact_centered_point` | 07-style point tactile replaced by contact-centred sparse patches | 17 cloud + 6 tactile | Does a force-centred local shape help? |
| 17 | `contact_centered_map` | 10-style map tactile replaced by the same force-centred patches and local attention | 17 cloud + 6 tactile | Does coupled force attention help after centring? |
| 18 | `base_pooled_relation` | 07 plus tactile–scene relation tokens | 17 cloud + 6 tactile + 6 relation | Does explicit local geometry conditioned on force help? |
| 19 | `base_map_matched_relation` | 10 plus tactile–scene relation tokens | 17 cloud + 6 tactile + 6 relation | Same relation test with coupled-map tactile encoding |

## 14 and 15: remove point cloud only

Both retain front RGB, wrist RGB, state, both 140-taxel base-frame force
fields, the original 6 tactile tokens and suffix injection.  They only omit
the fused 4096-point cloud and its 16 local plus 1 global visual tokens.

## 16 and 17: contact-centred tactile input

For each finger and each pre-action frame, force magnitude defines active
taxels.  The threshold is `max(5e-5, 0.25 * peak magnitude)`.  Four-connected
components are ranked by total force; the strongest two are retained.  Each
component produces an anchor at its force-weighted centre, then keeps the 16
taxels closest to that centre in fixed pad coordinates.  The 16 local inputs
are `[relative pad u,v, base Fx,Fy,Fz]`; force sign is retained.  Each anchor
also supplies a 10-D summary: resultant force, total magnitude, active-area
fraction, force-weighted spread, peak magnitude, pad-coordinate centre, and
validity.  A 13-D full-pad summary preserves signed/absolute/RMS/peak force
statistics and active fraction.

Each finger emits two anchor tokens plus one full-pad summary token, therefore
six tactile tokens in total.  The token also receives its base-frame anchor
position and finger identity.  A finger with no active taxels emits no local
anchor token but retains its full-pad token, so absence of contact is explicit.

16 uses masked mean/max point aggregation over the 16 taxels.  17 uses a
two-block coupled Fx/Fy/Fz self-attention module over the same taxels.  The
input is centred for shape matching, while absolute pad location remains
available in the token.  This is a *representation package*, not a perfectly
single-factor replacement for model 07: unlike 07's cross-finger pooling,
16/17 reserve two anchors per finger to keep bilateral contact centres clear.
Their active spatial-encoder parameter counts are matched to their respective
references: 16 has 254,400 parameters versus 07's 252,800 (0.6% difference),
and 17 has 1,103,248 versus 10's 1,103,168.  The small residual capacity
blocks used for that match do not add tokens or another sensor.

## 18 and 19: local tactile–scene relation tokens

The models retain their full original cloud and raw tactile encoders.  For
each finger, the force-weighted pad centre is a query.  From the 4096 cloud
points, the representation keeps up to 48 nearest points within 16 cm that
are on the tactile-facing side of the pad pair and farther than 3 mm from the
query.  A relation candidate carries `[candidate-query XYZ, distance, 13-D
finger force summary]`.  Three learned attention queries pool this set into
three tokens per finger, giving six relationship tokens.

The geometric filter removes points that are clearly behind the active pad.
The archived V3 point cloud has no robot or object semantic mask, so it is not
called full robot segmentation and it does not claim to identify the physical
USB–socket collision.  The audit page displays all local candidates and the
retained subset.  This is an observable local tactile–scene relation control.

## Required audit before training

Run locally:

```bash
cd /home/sai/zx/openpi-3d-tactile
bash sim/scripts/launch_contactworld_usb_contact_repr_audit.sh
```

It builds `data/contactworld_usb_v3_bilateral/spatial/contact_repr_v1`, checks
every episode and exactly reconstructs three frames per episode, then writes:

```text
runs/contactworld_usb_contact_repr_audit_YYYYMMDD/index.html
```

Review free movement, stable grasp, obstacle contact, insertion, and end-state
examples.  The training launcher repeats the sidecar verifier and sends an
actual transformed training item through every selected encoder before opening
any GPU worker.

## Training command after the audit is accepted

On the training server, generate or upload the reviewed
`spatial/contact_repr_v1` directory together with the USB V3 dataset.  Then:

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
export OPENPI_CONTACTWORLD_BASE_WEIGHTS=/workspace/mnt/sqzhang26/hf_weight/pi0_base/params
export OPENPI_CONTACTWORLD_GPUS=0,1,2,3,4,5
bash sim/scripts/launch_contactworld_usb_round2.sh cw3_usb_round2 --check-only
```

If preflight passes, remove `--check-only`.  A fresh label is required; all six
checkpoints and their logs are grouped under that label.
