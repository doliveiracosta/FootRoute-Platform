from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Place:
    name: str
    city: str
    state: str
    region: str
    lat: float
    lon: float

    @property
    def city_label(self) -> str:
        return f"{self.city}/{self.state}"


def load_places(path: Path) -> list[Place]:
    frame = pd.read_csv(path)
    required = {"name", "city", "state", "region", "lat", "lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Arquivo {path} sem coluna(s): {', '.join(sorted(missing))}")
    return [
        Place(
            name=str(row["name"]),
            city=str(row["city"]),
            state=str(row["state"]),
            region=str(row["region"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
        )
        for _, row in frame.iterrows()
    ]
