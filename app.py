from __future__ import annotations

from math import sqrt
from typing import Iterable

from .models import Place, haversine_km


REGION_COLORS = {
    "Norte": "#2ca25f",
    "Nordeste": "#f28e2b",
    "Centro-Oeste": "#756bb1",
    "Sudeste": "#4e79a7",
    "Sul": "#e15759",
}


def svg_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def project_points(places: Iterable[Place], width: int, height: int, margin: int) -> dict[str, tuple[float, float]]:
    place_list = list(places)
    min_lat = min(place.lat for place in place_list)
    max_lat = max(place.lat for place in place_list)
    min_lon = min(place.lon for place in place_list)
    max_lon = max(place.lon for place in place_list)

    positions = {}
    for place in place_list:
        x = margin + (place.lon - min_lon) / (max_lon - min_lon) * (width - 2 * margin)
        y = margin + (max_lat - place.lat) / (max_lat - min_lat) * (height - 2 * margin)
        positions[place.id] = (x, y)
    return positions


def route_svg(all_clubs: list[Place], route: list[Place]) -> str:
    width, height, margin = 1040, 660, 92
    positions = project_points(all_clubs, width, height, margin)
    route_ids = {place.id for place in route}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#111827" opacity="0.86" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<text x="36" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Rota otimizada</text>',
        '<text x="36" y="68" font-family="Arial" font-size="13" fill="#4b5563">Arestas direcionadas indicam a ordem operacional do percurso.</text>',
    ]

    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        x1, y1 = positions[origin.id]
        x2, y2 = positions[destination.id]
        distance = haversine_km(origin, destination)
        stroke_width = 2.0 + min(distance / 1200.0, 3.5)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#111827" stroke-width="{stroke_width:.2f}" opacity="0.58" marker-end="url(#arrow)">'
            f'<title>{idx}. {svg_escape(origin.name)} -> {svg_escape(destination.name)}: {distance:.0f} km</title></line>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lines.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" fill="#ffffff" stroke="#111827" stroke-width="1" />'
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="Arial" font-size="10" font-weight="700" fill="#111827">{idx}</text>'
        )

    for place in all_clubs:
        x, y = positions[place.id]
        color = REGION_COLORS.get(place.region, "#6b7280")
        radius = 11 if place.id in route_ids else 7
        opacity = "1" if place.id in route_ids else "0.35"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="#111827" stroke-width="1.4" opacity="{opacity}" />'
        )
        lines.append(
            f'<text x="{x + 14:.1f}" y="{y - 2:.1f}" font-family="Arial" font-size="12" '
            f'font-weight="700" fill="#111827" opacity="{opacity}">{svg_escape(place.name)}</text>'
        )
        lines.append(
            f'<text x="{x + 14:.1f}" y="{y + 13:.1f}" font-family="Arial" font-size="10" '
            f'fill="#4b5563" opacity="{opacity}">{svg_escape(place.city_label)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def nearest_capital_rows(clubs: list[Place], capitals: list[Place]) -> list[dict[str, object]]:
    rows = []
    for club in clubs:
        nearest = min(capitals, key=lambda capital: haversine_km(club, capital))
        rows.append(
            {
                "clube": club.name,
                "cidade_clube": club.city_label,
                "regiao_clube": club.region,
                "capital_referencia": nearest.name,
                "uf_capital": nearest.state,
                "regiao_capital": nearest.region,
                "distancia_capital_km": round(haversine_km(club, nearest), 1),
            }
        )
    return rows
