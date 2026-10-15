# ContactWorld USB event and sensor atlas: first interpretation

The completed run contains 95 train, 15 validation, and 10 test demonstrations. The report uses simulation contact and actual plug displacement to label events, independently of tactile input. Test contains 263 contact frames, 102 observed blocked-after-push frames, and 74 observed progress-after-push frames. First contact is counted once per test demonstration (10 events). The earlier webpage incorrectly counted 50 force threshold re-entries as “first contact”; the generator and report are corrected.

Open the [overview](/home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008/index.html) and the new [tactile-cluster video pairs](/home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008/cluster_pairs.html). The latter puts 12 videos into six side-by-side pairs, with synchronized play controls and the physical-state differences printed above each pair. It includes raw front/wrist RGB, both tactile previews, and six base-frame Fx/Fy/Fz maps. All video links and codecs were checked; a headless Chrome render was inspected.

## Findings from held-out demonstrations

| Question | Comparison on 10 test demonstrations | Interpretation |
|---|---:|---|
| Is the USB already contacting the socket? | State + prior actions BA 0.937; Fz-only 0.792; full tactile map-only 0.903; frozen dual-view RGB-only 0.910 | Fz alone is an imperfect contact detector. Force distribution contains more information, but much of contact status is already inferable from motion/state. |
| Was the previous downward push blocked? | State BA 0.849; state + signed Fz 0.872 | Paired episode bootstrap difference +0.023, 95% interval +0.010 to +0.039. Signed Fz provides a small, repeatable conditional signal in this diagnostic model. |
| Does the current executed downward push make progress? | State BA 0.835; state + absolute Fz 0.865 | Difference +0.030, paired 95% interval −0.016 to +0.074: promising but inconclusive. This evaluates only pushes that were actually executed. |
| Do Fx/Fy help these two labels? | State + Fx/Fy BA 0.838 for blocked, 0.796 for progress | No improvement with this coarse two-finger summary. This does **not** test whether spatial shear predicts the direction of a lateral correction. |
| Does RGB help after contact? | State + frozen RGB BA 0.848 for blocked, 0.749 for progress | This frozen ResNet feature did not improve these labels. It cannot establish that raw camera images are uninformative at millimeter scale. |

BA is balanced accuracy. Validation demonstrations selected regularization and decision thresholds; the test demonstrations were not used for fitting. Bootstrap resamples whole test demonstrations. These are exploratory estimates because there are only 10 independent test demonstrations and the label thresholds are analyst choices.

The signed Fx/Fy/Fz distributions of blocked and progressing frames overlap strongly. Event-aligned curves show an Fz rise around first socket contact, but there is no single signed Fz threshold that cleanly identifies contact mode or a safe descent. The task timing matters: tactile at frame *t* can reflect action *t−1*. The report therefore plots prior action, actual motion, and force on the same control-step axis.

## Tactile clusters

Clustering used only tactile input from simulator-confirmed socket-contact frames. Four views were fitted on train demonstrations: coarse force amplitude, normalized base-frame pattern, normalized sensor-local pattern, and base pattern with three prior force summaries. Test contact frames were assigned without consulting actions or outcomes. The clusters are associations, not ground-truth contact-state classes.

The coarse-amplitude cluster 5 is associated with stalled downward pushes: 2 progressing vs 32 blocked test pushes, from 9 demonstrations. Other clusters mix results. For example, normalized base-pattern cluster 1 has 23 progressing and 33 blocked pushes, so pattern identity alone does not determine what happens next. The video pairs intentionally show opposing outcomes under broadly similar tactile patterns and downward commands.

Suggested viewing order in the cluster page:

1. **Cluster 1**: source 11 frame 50 vs source 69 frame 45. These are the closest tactile-pattern pair among opposite outcomes with near pose/grasp: pattern distance 0.754, relative XY gap 0.69 mm, Z gap 0.87 mm, and downward-command gap 0.05 mm. Check the lateral command and the preceding frames before interpreting the difference.
2. **Cluster 5**: source 15 frame 86 vs source 69 frame 61. Relative XY gap 0.65 mm, Z gap 1.00 mm, grasp gap 0.94 mm; one advances about 2.21 mm while the other does not advance. The tactile distance is 1.303, so these are in the same broad class but not identical force fields.
3. **Cluster −1** is a useful cautionary example: the paired clips have a 7.58 mm axial and 11.32 mm grasp gap. They should **not** be interpreted as equivalent physical states just because both have weak force patterns.

The original webpage placed these 12 cluster videos after 24 event clips and a long statistics section, making them hard to find. The homepage now links to a dedicated two-column pair viewer at the top; the main page contains only event videos. A browser render and all media links were checked.

## What is not settled

This audit does not tell us which lateral or angular adjustment is effective at a given blocked state. Demonstration actions are not randomized, and touch may reflect the preceding action or sliding. The next decisive test is a small same-state simulator branch: restore one blocked state, hold prior history fixed, try a few candidate corrections and downward probes, and compare actual progress. The cluster page identifies useful candidate states, especially physically near yet opposite-outcome pairs.
