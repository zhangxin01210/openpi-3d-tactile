# UR/XHand glass task: safety interlock — NOT certified

## State of integration

`deploy/contact_safety.py` is a two-tier application-level monitor
for raw XHand tactile taxels and the arm's measured **TCP force (3D) and torque (3D)**. It runs before
each policy action and once again after a potentially blocking Pi0 inference.
Moderate tactile changes may discard remaining actions and request a new Pi0 chunk ONLY when explicitly enabled. Hard limits latch and terminate the run; they never replan-and-continue. `robot.stop()` is called on the
fault path; its physical stopping distance and behavior must be tested on
a non-fragile surrogate. Independent UR protective-stop, conservative speed
and acceleration limits, and a physically accessible hardware E-stop remain
required. This is not a deterministic real-time or safety-certified controller.

**Actuation remains BLOCKED pending physical safety validation:** The
standalone robot script now passes `record_tcp_force_torque=True` to
`UR7eConfig`; on this LeRobot version, combined observations are expected to
include `arm_tcp_force` shape (3,) [N] and `arm_tcp_torque` shape (3,)
[Nm]. The guard checks both independently and fails closed if either is
missing/malformed. Confirm these facts with the SECOND read-only audit before
any actuator command. The exact RTDE force-bias, reference frame, timestamps
and protective-stop behavior still require independent physical validation.
The 1972-state still contains hand motor **current** labelled
`hand_joint_i.torque`, NOT measured UR joint torque.

## Data survey — never automatic limits

Run on computer with canonical original dataset, **read-only**:

```bash
PYTHONPATH="$PWD/src:$PWD" "$OLD/.venv/bin/python" \
 scripts/deploy/audit_glass_contact_limits.py \
 --dataset "$OLD/data/1018_peg_all" \
 --output "$PWD/output/deploy/qa/glass_tactile_distribution.json"
```

It reports 5 per-finger max-taxel vector norm distributions (p95/p99/p99.9/
max), change per consecutive frame and sample provenance. All values remain
XHand raw units. Do not copy max or arbitrary percentages into actuation
limits. Historical data may include glass destruction, sensor saturation,
noise spikes and successes that still stressed equipment. Determine limits
with independent sensor calibration, no-glass surrogates, slow controlled
contacts and hardware protective stop. Prefer explicitly per-finger
thresholds with instrumented inspection; sudden *loss* of contact may be
unsafe too, so the step-change check is symmetric.

## Explicit configuration schema, NO numeric defaults

Create a locally reviewed JSON file. This is a SCHEMA EXAMPLE, values must
be deliberately determined on the physical system:

```json
{
  "schema_version": 1,
  "reviewed_for_glass_task": false,
  "tactile_peak_limits_raw": [null, null, null, null, null],
  "tactile_emergency_step_change_limits_raw": [null, null, null, null, null],
  "tactile_replan_step_change_limits_raw": [null, null, null, null, null],
  "replan_enabled": false,
  "replan_min_interval_s": null,
  "arm_tcp_force_norm_limit_n": null,
  "arm_tcp_torque_norm_limit_nm": null,
  "max_observation_gap_s": null
}
```

This intentionally fails validation until operators establish and review
real sensor-specific limits. The measured UR interface supplies two explicit 3-element
arrays (arm_tcp_force in N, arm_tcp_torque in Nm), not a synthetic 6-element key. A single raw force peak is a per-taxel
norm, not a finger resultant. An observation gap trips; large blocking
inference can cause a stale-gap trip, which is safer than executing a stale
chunk. Nonfinite/missing sensor data trips. `--contact-safety-config` is
mandatory when `--dry-run` is absent.

## Release checks

1. Confirm that the SDK exposes arm force/torque at every commanded step;
   verify its measurements and timestamps under non-fragile contact.
2. Verify tactile arrays against the actual live 1972-state layout and
   that resetting zero under load cannot suppress dangerous contact.
3. Replay recorded normal and abnormal samples and show that over-limit,
   sudden increase, sudden loss, sensor missing, delayed inference, and
   sensor staleness each produce an immediate latched command inhibit.
4. Confirm no command is queued after a trip, including at chunk boundaries;
   measure total sensor-to-stop latency and robot stopping distance.
5. Validate UR controller protective stop and hardware E-stop independently;
   use non-glass surrogates before carefully controlled glass tasks.
6. The perception NPZ producer and Pi0 dry run are NOT YET connected to the
   robot control client's websocket/action path. Do not actuate with the old
   client expecting the new object geometry tokens.

A 15 Hz application loop provides at most one observation every ~67 ms when
running on schedule; impact can occur between samples. The 16-action Pi0 chunk
is NOT a permitted blind 16-step motion. Hardware contact protection must
cover the gaps.

## Paper evaluation protocol

Keep `replan_enabled=false` for the primary fixed-chunk comparison and use identical hard-stop protection on every method. Event-replanning is a distinct, disclosed policy-scheduling ablation, enabled identically for all compared methods and with equal inference budget reporting. It is not a certified safety action. A transient contact event never overrides a hard stop. No threshold has been validated against breakage force.

## October 28 live audit updates

The first read-only audit yielded five 640x480 RGB pairs and front 480x640
uint16 depth arrays of **unknown coordinate frame**. UR TCP force and torque
were absent because `record_tcp_force_torque` had been left off. The next
revision now turns it on in `build_robot()`; run a new *read-only* audit and
check both fields are present as 3-vectors before treating the interface as
available. Every one of the five XHand taxel arrays was entirely zero, not
only its 5 peak summaries. That cannot distinguish genuinely unloaded hand
from an SDK stale/placeholder packet. In particular, XHand's previous-value
fallback / zero-filled sensor failure must be identified and reported as a
separate sensor-health signal; a finite 1972-vector is insufficient to assert
safety monitor freshness. Do NOT authorize contact/glass control until this is
resolved. The calibration and protection thresholds are still NOT set.
