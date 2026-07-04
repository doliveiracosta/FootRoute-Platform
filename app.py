from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import asin, cos, radians, sin, sqrt

from footroute.models import Place

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Place, b: Place) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [a.lat, a.lon, b.lat, b.lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def route_distance(route: list[Place]) -> float:
    return sum(haversine_km(a, b) for a, b in zip(route, route[1:]))


def distance_matrix(places: list[Place]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for origin in places:
        row: dict[str, object] = {"Clube": origin.name}
        for destination in places:
            row[destination.name] = round(haversine_km(origin, destination), 1)
        rows.append(row)
    return rows


def route_rows(route: list[Place], long_trip_km: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        distance = haversine_km(origin, destination)
        rows.append(
            {
                "ordem": order,
                "origem": origin.name,
                "destino": destination.name,
                "cidade_origem": origin.city_label,
                "cidade_destino": destination.city_label,
                "regiao_origem": origin.region,
                "regiao_destino": destination.region,
                "distancia_km": round(distance, 1),
                "interregional": origin.region != destination.region,
                "viagem_longa": distance >= long_trip_km,
            }
        )
    return rows


def summary_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    distances = [float(row["distancia_km"]) for row in rows]
    return {
        "total_km": sum(distances),
        "trechos": len(rows),
        "viagens_longas": sum(1 for row in rows if bool(row["viagem_longa"])),
        "interregionais": sum(1 for row in rows if bool(row["interregional"])),
        "maior_trecho_km": max(distances, default=0.0),
    }


def _two_opt(route: list[Place], fixed_start: bool = True, fixed_end: bool = False) -> list[Place]:
    if len(route) < 4:
        return route
    best = route[:]
    improved = True
    while improved:
        improved = False
        start_i = 1 if fixed_start else 0
        end_k = len(best) - (1 if fixed_end else 0)
        for i, k in combinations(range(start_i, end_k), 2):
            if k - i < 2:
                continue
            candidate = best[:i] + list(reversed(best[i:k])) + best[k:]
            if route_distance(candidate) + 1e-9 < route_distance(best):
                best = candidate
                improved = True
                break
        if improved:
            continue
    return best


def heuristic_route(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    remaining = destinations[:]
    route = [start]
    current = start
    while remaining:
        nxt = min(remaining, key=lambda place: haversine_km(current, place))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    if return_to_start:
        route.append(start)
        route = _two_opt(route, fixed_start=True, fixed_end=True)
        if route[-1] != start:
            route.append(start)
    else:
        route = _two_opt(route, fixed_start=True, fixed_end=False)
    return route, route_distance(route)


def held_karp(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    places = [start] + destinations
    n = len(places)
    if n == 1:
        return [start], 0.0
    if len(destinations) > 12:
        return heuristic_route(start, destinations, return_to_start)

    dist = [[haversine_km(places[i], places[j]) for j in range(n)] for i in range(n)]

    @lru_cache(maxsize=None)
    def dp(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
        if mask == (1 << last):
            return dist[0][last], (0, last)
        prev_mask = mask & ~(1 << last)
        best_cost = float("inf")
        best_path: tuple[int, ...] = ()
        for prev in range(1, n):
            if prev_mask & (1 << prev):
                cost, path = dp(prev_mask, prev)
                candidate = cost + dist[prev][last]
                if candidate < best_cost:
                    best_cost = candidate
                    best_path = path + (last,)
        return best_cost, best_path

    full_mask = sum(1 << i for i in range(1, n))
    best_cost = float("inf")
    best_path: tuple[int, ...] = ()
    for last in range(1, n):
        cost, path = dp(full_mask, last)
        if return_to_start:
            cost += dist[last][0]
        if cost < best_cost:
            best_cost = cost
            best_path = path

    route = [places[index] for index in best_path]
    if return_to_start:
        route.append(start)
    return route, best_cost
