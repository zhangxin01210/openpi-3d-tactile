# UR/XHand glass task: safety interlock — NOT certified

## State of integration

`deploy/contact_safety.py` is an application-level fail-closed monitor
for raw XHand tactile taxels and the arm's **6D TCP wrench**. It runs before
each policy action and once again after a potentially blocking Pi0 inference.
A trip latches and terminates the run; it never requests a new policy chunk
and then continues pushing against glass. `robot.stop()` is called on the
fault path; its physical stopping distance and behavior must be tested on
a non-fragile surrogate. Independent UR protective-stop, conservative speed
and acceleration limits, and a physically accessible hardware E-stop remain
required. This is not a deterministic real-time or safety-certified controller.

**Actuation remains BLOCKED:** No real six-axis arm wrench observation key
has yet been confirmed in `UR7eXHand.get_observation()`. Existing 1972-state
array includes hand torque but NOT arm torque. The client fails closed
when the configured real arm-wrench key is absent. Do not invent a key or
use a hand torque field as the arm wrench. Its frame, zeroing, units and
freshness must be verified against RTDE / UR SDK and device manual; the
application monitors force and torque norms in the supplied frame, *not*
the six UR joint torques. If only joint torques are exposed, define and
validate a separate explicit interface before enabling actuation.

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
  "arm_wrench_observation_key": "REAL_VERIFIED_ARM_WRENCH_KEY",
  "tactile_peak_limits_raw": [null, null, null, null, null],
  "tactile_step_change_limits_raw": [null, null, null, null, null],
  "arm_tcp_force_norm_limit_n": null,
  "arm_tcp_torque_norm_limit_nm": null,
  "max_observation_gap_s": null
}
```

This intentionally fails validation until operators establish and review
real sensor-specific limits. The measured wrench needs a verified 6-element
array [Fx,Fy,Fz,Tx,Ty,Tz] in N and N m. A single raw force peak is a per-taxel
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
