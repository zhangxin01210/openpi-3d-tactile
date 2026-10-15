# ContactWorld USB event and sensor atlas

Run from the repository root:

```bash
bash sim/scripts/launch_contactworld_event_sensor_atlas.sh
```

The launcher detaches, so closing the terminal does not interrupt the job. It reads the existing bilateral LeRobot dataset, the history audit cache, and the local replay metadata. It does not download anything or require a GPU. It writes a separate directory:

```text
/home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008/
├── launch.log
├── index.html
├── cluster_pairs.html
├── events.json
├── probes.json
├── clusters.json
├── cases.json
├── event_aligned_counts.json
├── predictions/
├── figures/
├── cases/
└── complete.json
```

To follow and check completion:

```bash
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008/launch.log
cat /home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008/complete.json
```

The last log line should start with `DONE`. `complete.json` is written only after the HTML report and all selected videos have been generated. The same launch command can be run again after interruption. An output directory is locked during a run. To change it, set `CW_EVENT_ATLAS_OUTPUT` before the command.

## What is compared

- Physical event labels use replay contact/success records, executed commands, and actual plug displacement. Touch is not used to define an event.
- Probe models fit on complete train demonstrations, select regularization and threshold on separate validation demonstrations, and report test results on separate demonstrations. Inputs compare robot state and action history, frozen dual-view RGB features, scalar Fz, signed Fz, Fx/Fy, six-axis force summary, the full two-finger force map, temporal force history, and RGB plus force map.
- Four tactile clusterings compare amplitude, robot-base spatial pattern, sensor-local spatial pattern, and pattern with a short force history. Their centers use train contact frames only. Test frames are assigned without consulting their action or outcome.
- The overview links to a dedicated side-by-side cluster-pair viewer with synchronized playback. For each event and selected cluster pair, videos show original front/wrist RGB, left/right tactile preview, both fingers' Fx/Fy/Fz grids, and adjacent action/motion timelines. Cluster summaries separately report the subset where a downward command was actually executed.

The `blocked_after_push` label means a previously executed downward command failed to move the plug down in the next observation. The `progress_when_pushed` label is only evaluated where a downward command was actually executed. Borderline displacement is excluded. `recovery_candidate` is an observational label based on later movement; it is not proof that a particular correction caused recovery. The replay cache ends four frames before each episode ends, so late completion may be absent from frame-level statistics. The RGB probe uses frozen ResNet features and cannot rule out fine visual information missed by that representation. Per-frame results are exploratory; episode bootstrap intervals show the limited number of independent demonstrations.

If the report identifies states where touch or vision are ambiguous, the next experiment is a small set of controlled action branches from the same restored simulation state. This launcher only runs the offline audit.
