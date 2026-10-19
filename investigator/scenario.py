"""Load the agent-side half of a scenario (report + environment + tool faults).

Ground truth lives in a separate directory that this package never reads.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dataset.schema import Scenario

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "dataset" / "scenarios"


def scenario_ids() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))


def load_scenario(scenario: str | Path) -> Scenario:
    """Load by id (e.g. the file stem) or by explicit path."""
    path = Path(scenario)
    if not path.suffix:
        path = SCENARIOS_DIR / f"{scenario}.yaml"
    return Scenario.model_validate(yaml.safe_load(path.read_text()))
