from __future__ import annotations

from footroute.models import Place
from footroute.optimization import haversine_km


def nearest_capital_rows(clubs: list[Place], capitals: list[Place]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for club in clubs:
        capital = min(capitals, key=lambda item: haversine_km(club, item))
        rows.append(
            {
                "clube": club.name,
                "cidade_clube": club.city_label,
                "regiao_clube": club.region,
                "capital_referencia": capital.city,
                "uf_capital": capital.state,
                "regiao_capital": capital.region,
                "distancia_capital_km": round(haversine_km(club, capital), 1),
            }
        )
    return rows


def route_svg(clubs: list[Place], route: list[Place]) -> str:
    if not clubs:
        return ""
    width = 980
    height = 620
    margin = 50
    min_lat = min(place.lat for place in clubs)
    max_lat = max(place.lat for place in clubs)
    min_lon = min(place.lon for place in clubs)
    max_lon = max(place.lon for place in clubs)

    def scale_x(lon: float) -> float:
        if max_lon == min_lon:
            return width / 2
        return margin + (lon - min_lon) / (max_lon - min_lon) * (width - 2 * margin)

    def scale_y(lat: float) -> float:
        if max_lat == min_lat:
            return height / 2
        return height - margin - (lat - min_lat) / (max_lat - min_lat) * (height - 2 * margin)

    points = {place.name: (scale_x(place.lon), scale_y(place.lat)) for place in clubs}
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="620" xmlns="http://www.w3.org/2000/svg">',
        '<rect x="0" y="0" width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        '<text x="28" y="36" font-size="22" font-family="Arial" font-weight="700" fill="#111827">Grafo da rota otimizada</text>',
        '<text x="28" y="62" font-size="13" font-family="Arial" fill="#475569">Vértices = clubes/cidades-sede; arestas = deslocamentos selecionados</text>',
    ]

    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        x1, y1 = points[origin.name]
        x2, y2 = points[destination.name]
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#2563eb" stroke-width="3" stroke-linecap="round" opacity="0.8"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        svg_parts.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" fill="#1d4ed8" opacity="0.92"/>'
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" font-size="10" text-anchor="middle" font-family="Arial" fill="white">{idx}</text>'
        )

    route_names = {place.name for place in route}
    for place in clubs:
        x, y = points[place.name]
        selected = place.name in route_names
        radius = 8 if selected else 5
        fill = "#ef4444" if selected else "#64748b"
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="white" stroke-width="2"/>')
        svg_parts.append(
            f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-size="12" font-family="Arial" fill="#0f172a">{place.name}</text>'
        )
        svg_parts.append(
            f'<text x="{x + 10:.1f}" y="{y + 7:.1f}" font-size="10" font-family="Arial" fill="#64748b">{place.city_label}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)
