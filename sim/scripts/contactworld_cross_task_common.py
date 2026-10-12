"""Shared immutable task contract for the ContactWorld cross-task study.

This file deliberately contains no Isaac Gym imports.  Download, source export,
replay, LeRobot export, training and evaluation all read the same contract so a
task cannot silently acquire a different action shape or instruction midway.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    key: str
    source_group: str
    archive: str | None
    hydra_config: str
    task_class: str
    action_dim: int
    prompt: str
    output_dataset: str


TASKS = {
    "peg": TaskSpec(
        key="peg", source_group="insertion_peg", archive=None,
        hydra_config="peg", task_class="TacSLTaskPeg", action_dim=6,
        prompt="Insert the peg into the matching hole.",
        output_dataset="contactworld_peg_v3_bilateral",
    ),
    "barbed_flat": TaskSpec(
        key="barbed_flat", source_group="disassembly_barbed_flat",
        archive="disassembly_barbed_flat_dataset.tar.gz",
        hydra_config="barbed_flat", task_class="TacSLTaskBarbedFlat", action_dim=7,
        prompt="Remove the flat barbed plug from its socket.",
        output_dataset="contactworld_barbed_flat_v3_bilateral",
    ),
    "valve": TaskSpec(
        key="valve", source_group="screwing_valve",
        archive="screwing_valve_dataset.tar.gz",
        hydra_config="valve", task_class="TacSLTaskValve", action_dim=7,
        prompt="Turn the valve to complete the screwing task.",
        output_dataset="contactworld_valve_v3_bilateral",
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as error:
        raise ValueError("Unknown task %r; choices: %s" % (name, ", ".join(TASKS))) from error
