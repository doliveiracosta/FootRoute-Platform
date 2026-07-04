from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import csv


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    city: str
    state: str
    region: str
    lat: float
    lon: float

    @property
    def city_label(self) -> str:
        return f"{self.city}-{self.state}"


def load_places(path: Path) -> list[Place]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            Place(
                id=row["id"],
                name=row["nome"],
                city=row["cidade"],
                state=row["uf"],
                region=row["regiao"],
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
            )
            for row in reader
        ]


def haversine_km(a: Place, b: Place) -> float:
    radius = 6371.0
    lat1, lon1 = radians(a.lat), radians(a.lon)
    lat2, lon2 = radians(b.lat), radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(h))
