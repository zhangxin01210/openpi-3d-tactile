# XHand final V2 training

All three datasets use `spatial/v1`, `object_geometry/v1`, and
`hand_keypoints/v1`. Full requires all three; ST requires spatial only; RGB
requires no sidecar. The dataset QA figures supplied for 1018, 1028 and 1030
are 70/32,343, 100/26,485 and 111/34,963 episodes/frames respectively. Preflight
checks the actual files and indexed joins again; it does not regenerate data.

| Config | Dataset | Base | Spatial tokens |
|---|---|---|---:|
| `pi0_xhand_1028_press_button_rgb_v2` | 1028_press_button | π0 | 0 |
| `pi05_xhand_1028_press_button_rgb_v2` | 1028_press_button | π0.5 | 0 |
| `pi0_xhand_1028_press_button_st_v2` | 1028_press_button | π0 | 58 |
| `pi0_xhand_1028_press_button_full_v2` | 1028_press_button | π0 | 68 |
| `pi0_xhand_1030_yellow_socket_rgb_v2` | 1030_yellow_socket | π0 | 0 |
| `pi05_xhand_1030_yellow_socket_rgb_v2` | 1030_yellow_socket | π0.5 | 0 |
| `pi0_xhand_1030_yellow_socket_st_v2` | 1030_yellow_socket | π0 | 58 |
| `pi0_xhand_1030_yellow_socket_full_v2` | 1030_yellow_socket | π0 | 68 |
| `pi0_xhand_1018_peg_all_st_v2` | 1018_peg_all | π0 | 58 |
| `pi0_xhand_1018_peg_all_full_v2` | 1018_peg_all | π0 | 68 |

The scene branch uses masked 256/24 then 32/16 FPS/KNN with mean/max pooling,
the existing raw-point global token, position and modality embeddings. The
tactile branch retains taxel MLP and per-finger EdgeConv and pools four fixed
link-local geometry regions plus one whole-finger region. HO queries the
manipulated cloud from the 18 exported FK anchors and pools each finger's own
anchors. OO retains 8×8 object regions and five learned queries. HO and OO use
only relative XYZ and distance as relation features; stored metadata is not
modified. Full and ST use the same V2 branches and existing suffix router.

The palm anchor is `palm/frame_origin` on `right_hand_link`. It is a palm frame
reference, not a finger joint or tip, and is excluded from finger token pooling.
The remaining anchors and their associated child or tip links are:

| Finger | Anchor → link |
|---|---|
| thumb | `right_hand_thumb_bend_joint` → `right_hand_thumb_bend_link`; `right_hand_thumb_rota_joint1` → `right_hand_thumb_rota_link1`; `right_hand_thumb_rota_joint2` → `right_hand_thumb_rota_link2`; `right_hand_thumb_rota_tip` → same-named link |
| index | `right_hand_index_bend_joint` → `right_hand_index_bend_link`; `right_hand_index_joint1` → `right_hand_index_rota_link1`; `right_hand_index_joint2` → `right_hand_index_rota_link2`; `right_hand_index_rota_tip` → same-named link |
| middle | `right_hand_mid_joint1` → `right_hand_mid_link1`; `right_hand_mid_joint2` → `right_hand_mid_link2`; `right_hand_mid_tip` → same-named link |
| ring | `right_hand_ring_joint1` → `right_hand_ring_link1`; `right_hand_ring_joint2` → `right_hand_ring_link2`; `right_hand_ring_tip` → same-named link |
| pinky | `right_hand_pinky_joint1` → `right_hand_pinky_link1`; `right_hand_pinky_joint2` → `right_hand_pinky_link2`; `right_hand_pinky_tip` → same-named link |

From the A800 checkout root, set `OPENPI_PI0_BASE_WEIGHTS` and
`OPENPI_PI05_BASE_WEIGHTS` if the defaults under
`/workspace/mnt/sqzhang26/hf_weight/{pi0_base,pi05_base}/params` differ. Set
`OPENPI_1018_FRONT_PROFILE`, `OPENPI_1028_FRONT_PROFILE`, and/or
`OPENPI_1030_FRONT_PROFILE` to each dataset's **original** profile file when
required by its manifest. A file with equivalent JSON but a different SHA256 is
not a replacement. Full preflight rejects missing object calibration provenance
or a mismatching profile. Investigate provenance rather than disabling checks.

Each dataset shares `assets/xhand/<dataset>_unified_v1/norm_stats.json` across
its configurations. The native π0 pipeline uses mean/std; the native π0.5
pipeline uses quantiles from the same statistics. If missing, compute the stats
using the existing script; select the dataset's RGB config, or the 1018 ST
config, for example:

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_norm_stats.py --config-name pi0_xhand_1028_press_button_rgb_v2
```

Run preflight before loading large models, then smoke (short training writes to
a temporary, independent checkpoint directory):

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/preflight_final_v2.py --all-configs --json /tmp/final_v2_preflight.json
PYTHONPATH=src .venv/bin/python scripts/experiments/smoke_final_v2.py --all-configs --check all
```

Use `--config CONFIG --check data|model|train|all` to isolate a configuration.
Review preflight errors per configuration: `--all-configs` continues after
individual failures. The following commands perform preflight again and then use
the original trainer in the requested order:

```bash
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_rgb_v2
bash scripts/experiments/run_final_v2.sh pi05_xhand_1028_press_button_rgb_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_full_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_rgb_v2
bash scripts/experiments/run_final_v2.sh pi05_xhand_1030_yellow_socket_rgb_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_full_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1018_peg_all_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1018_peg_all_full_v2
```

`bash scripts/experiments/run_final_v2.sh all` runs that order sequentially.
Resume one existing experiment with `bash scripts/experiments/run_final_v2.sh
CONFIG --resume`. The default experiment name is `final_v2`; override with
`OPENPI_FINAL_V2_EXP_NAME`. The trainer refuses an existing directory unless
resume is requested. Check `logs/final_v2/` for commit, config, sidecars and
weight source, and `checkpoints/CONFIG/EXPERIMENT/` for saved steps.

Local checks: V2 module, configuration, simulated sidecar join and router
tests passed (10 tests); the online, configuration, join and relation suite
passed after integration (11 tests); existing object geometry/config tests
passed (7 tests), and CPU online/DataLoader tests passed (6 tests, excluding
the real-data and multiworker cases). An attempted multiworker DataLoader test
failed when the local CUDA backend could not initialize; it is not a passed
check. Configuration preflight resolved all ten configurations locally but
reported missing local data, assets and weights.

Local tests use synthetic inputs and actual repository geometry only. Real
sidecars, calibrated camera metadata, pretrained weights, A800 memory use,
loss/backward/update and checkpoint restoration require cloud verification.
Do not treat configuration registration or local tests as a successful cloud
smoke run. Object pose estimates remain subject to measurement error; HO
distances are not verified physical contact distances.
