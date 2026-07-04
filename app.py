from __future__ import annotations

from pathlib import Path
import sys
import html
from math import hypot

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footroute.models import Place, load_places  # noqa: E402
from footroute.optimization import (  # noqa: E402
    held_karp,
    heuristic_route,
    route_distance,
    route_rows,
    summary_metrics,
)


DATA_DIR = ROOT / "data"
LONG_TRIP_DEFAULT = 1500.0


@st.cache_data
def load_data() -> tuple[list[Place], list[Place]]:
    clubs = load_places(DATA_DIR / "clubes.csv")
    capitals = load_places(DATA_DIR / "capitais_brasil.csv")
    return clubs, capitals


def format_km(value: float) -> str:
    return f"{value:,.1f} km".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percent(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


def bool_label(value: bool) -> str:
    return "Sim" if value else "Não"


def display_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "ordem": "Ordem",
            "origem": "Origem",
            "destino": "Destino",
            "cidade_origem": "Cidade origem",
            "cidade_destino": "Cidade destino",
            "regiao_origem": "Região origem",
            "regiao_destino": "Região destino",
            "distancia_km": "Distância (km)",
            "interregional": "Inter-regional",
            "viagem_longa": "Viagem longa",
        }
    )
    frame["Inter-regional"] = frame["Inter-regional"].map(bool_label)
    frame["Viagem longa"] = frame["Viagem longa"].map(bool_label)
    return frame


def baseline_route(start: Place, destinations: list[Place], return_to_start: bool) -> list[Place]:
    route = [start] + destinations
    if return_to_start:
        route.append(start)
    return route


def solve_route(
    algorithm: str,
    start: Place,
    destinations: list[Place],
    return_to_start: bool,
) -> tuple[list[Place], float]:
    if algorithm.startswith("Exato"):
        return held_karp(start, destinations, return_to_start)
    return heuristic_route(start, destinations, return_to_start)


def visible_sequence(route: list[Place], start: Place) -> str:
    names = [place.name for place in route if place.name != start.name]
    return " → ".join(names) if names else "Nenhum clube selecionado."


def objective_terms(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "Z = 0"
    terms = []
    for row in rows:
        origin = str(row["origem"])
        destination = str(row["destino"])
        distance = float(row["distancia_km"])
        terms.append(f"{distance:.1f}·x_{{{origin},{destination}}}")
    return "Z = " + " + ".join(terms)


def _project_points(places: list[Place], width: int, height: int, top_pad: int = 110, side_pad: int = 44):
    lons = [p.lon for p in places]
    lats = [p.lat for p in places]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)

    usable_w = width - 2 * side_pad
    usable_h = height - top_pad - side_pad

    coords = {}
    for p in places:
        x = side_pad + ((p.lon - min_lon) / lon_span) * usable_w
        y = top_pad + (1 - (p.lat - min_lat) / lat_span) * usable_h
        coords[p.name] = (x, y)
    return coords


def _city_key(place: Place) -> str:
    return getattr(place, "city_label", f"{place.city}/{place.state}")


def _jitter_city_points(points: dict[str, tuple[float, float]], places: list[Place]) -> dict[str, tuple[float, float]]:
    """Separa visualmente clubes com a mesma cidade-sede."""
    grouped: dict[str, list[Place]] = {}
    for place in places:
        grouped.setdefault(_city_key(place), []).append(place)

    adjusted = dict(points)
    patterns = [
        (0, 0),
        (-16, -14),
        (16, -14),
        (-16, 14),
        (16, 14),
        (0, -25),
        (0, 25),
        (-28, 0),
        (28, 0),
    ]

    for group in grouped.values():
        if len(group) <= 1:
            continue
        cx = sum(points[p.name][0] for p in group) / len(group)
        cy = sum(points[p.name][1] for p in group) / len(group)
        for idx, place in enumerate(sorted(group, key=lambda item: item.name)):
            dx, dy = patterns[idx % len(patterns)]
            adjusted[place.name] = (cx + dx, cy + dy)
    return adjusted


def _dense_groups(points: dict[str, tuple[float, float]], places: list[Place], threshold: float = 88.0):
    """Agrupa rótulos que ficariam sobrepostos, mesmo que estejam em cidades próximas."""
    remaining = set(place.name for place in places)
    by_name = {place.name: place for place in places}
    groups: list[list[Place]] = []

    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        sx, sy = points[seed]
        group_names = [seed]

        changed = True
        while changed:
            changed = False
            for name in list(remaining):
                x, y = points[name]
                if any(hypot(x - points[g][0], y - points[g][1]) <= threshold for g in group_names):
                    group_names.append(name)
                    remaining.remove(name)
                    changed = True

        groups.append([by_name[name] for name in sorted(group_names)])

    return groups


def _callout_position(group: list[Place], base_points: dict[str, tuple[float, float]], width: int, height: int, order: int):
    xs = [base_points[p.name][0] for p in group]
    ys = [base_points[p.name][1] for p in group]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    # Distribuição determinística dos blocos de rótulos por quadrante.
    if cx > width * 0.62 and cy < height * 0.55:
        tx, ty = width - 330, 120 + order * 104
    elif cx > width * 0.62:
        tx, ty = width - 330, height - 210 - order * 104
    elif cx < width * 0.35 and cy > height * 0.55:
        tx, ty = 70, height - 210 - order * 104
    else:
        tx, ty = 70, 120 + order * 104

    tx = max(40, min(width - 340, tx))
    ty = max(105, min(height - 120, ty))
    return tx, ty, cx, cy


def route_svg(clubs: list[Place], route: list[Place], total_distance_label: str) -> str:
    width, height = 1280, 780
    raw_points = _project_points(clubs, width, height)
    points = _jitter_city_points(raw_points, clubs)
    selected_names = {p.name for p in route}

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" ',
        'style="background:#f3f4f6;border-radius:12px;font-family:Arial, Helvetica, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" ry="12" fill="#f3f4f6"/>',
        '<text x="24" y="36" font-size="20" font-weight="700" fill="#0f172a">Grafo da rota otimizada</text>',
        '<text x="24" y="62" font-size="13" fill="#334155">Vértices = clubes/cidades-sede</text>',
        '<text x="24" y="80" font-size="13" fill="#334155">Arestas = deslocamentos selecionados</text>',
        '<rect x="950" y="18" width="300" height="54" rx="10" fill="#ffffff" opacity="0.92" stroke="#cbd5e1" stroke-width="1"/>',
        '<text x="968" y="40" font-size="13" font-weight="700" fill="#334155">Distância total</text>',
        f'<text x="968" y="62" font-size="22" font-weight="700" fill="#0f172a">{html.escape(total_distance_label)}</text>',
    ]

    if len(route) >= 2:
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            x1, y1 = points[a.name]
            x2, y2 = points[b.name]
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#3b82f6" stroke-width="3.2" stroke-linecap="round" opacity="0.95"/>'
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            svg.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="12" fill="#2563eb" opacity="0.95"/>')
            svg.append(
                f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" '
                'font-size="11" font-weight="700" fill="#ffffff">'
                f'{i + 1}</text>'
            )

    for p in clubs:
        x, y = points[p.name]
        in_route = p.name in selected_names
        radius = 9 if in_route else 6
        fill = "#ef4444" if in_route else "#94a3b8"
        stroke = "#ffffff"
        stroke_w = 2.8 if in_route else 1.6
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )

    svg.append('</svg>')
    return ''.join(svg)

st.set_page_config(page_title="FootRoute", layout="wide")

clubs, capitals = load_data()
clubs_by_name = {club.name: club for club in clubs}
club_names = list(clubs_by_name)

st.title("FootRoute")
st.caption("Painel de otimização de rotas logísticas entre clubes de futebol.")

with st.sidebar:
    st.header("Configuração")
    default_index = club_names.index("Flamengo") if "Flamengo" in club_names else 0
    start_name = st.selectbox("Clube de origem", club_names, index=default_index)
    available = [name for name in club_names if name != start_name]
    selected_names = st.multiselect("Clubes a visitar", available, default=available)
    return_to_start = st.checkbox("Retornar ao clube de origem", value=True)
    algorithm = st.radio(
        "Algoritmo",
        ["Exato (Held-Karp)", "Heurístico (vizinho mais próximo + 2-opt)"],
        index=0,
    )
    long_trip_km = st.slider("Limiar de viagem longa (km)", 500, 3000, int(LONG_TRIP_DEFAULT), 100)

start = clubs_by_name[start_name]
destinations = [clubs_by_name[name] for name in selected_names]
route, total_distance = solve_route(algorithm, start, destinations, return_to_start)
rows = route_rows(route, long_trip_km=float(long_trip_km))
metrics = summary_metrics(rows)

baseline = baseline_route(start, destinations, return_to_start)
baseline_distance = route_distance(baseline)
gain = 0.0 if baseline_distance == 0 else (baseline_distance - total_distance) / baseline_distance * 100

total_distance_label = format_km(metrics["total_km"])

tab_route, tab_legs, tab_model = st.tabs(["Rota", "Trechos", "Modelo"])

with tab_route:
    components.html(route_svg(clubs, route, total_distance_label), height=780, scrolling=False)
    st.subheader("Sequência recomendada")
    st.write(visible_sequence(route, start))
    st.download_button(
        "Baixar rota em CSV",
        data=display_rows(rows).to_csv(index=False).encode("utf-8-sig"),
        file_name="footroute_rota_otimizada.csv",
        mime="text/csv",
    )

with tab_legs:
    st.dataframe(display_rows(rows), width="stretch", hide_index=True)

with tab_model:
    st.markdown("### Formulação resumida")
    st.write(
        "Considere um grafo ponderado, em que cada vértice representa um clube "
        "e cada aresta representa o deslocamento entre duas cidades-sede."
    )
    st.latex(r"G=(V,E)")
    st.write(
        "O peso associado a cada aresta é a distância geodésica aproximada entre "
        "os clubes i e j."
    )
    st.latex(r"d_{ij}=\text{distância geodésica entre os clubes } i \text{ e } j")
    st.write("Para uma rota representada por uma sequência de visitação:")
    st.latex(r"\pi=(\pi_0,\pi_1,\ldots,\pi_n)")
    st.write("A função objetivo é minimizar a distância total percorrida:")
    st.latex(r"\min Z = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")
    if return_to_start:
        st.write("Como há retorno à origem, a rota é tratada como um ciclo:")
        st.latex(r"Z_{\text{ciclo}} = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}} + d_{\pi_n,\pi_0}")
    else:
        st.write("Como não há retorno à origem, a rota é tratada como caminho aberto:")
        st.latex(r"Z_{\text{aberto}} = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")
    st.markdown("### Função objetivo numérica da rota atual")
    st.code(objective_terms(rows), language="text")
    st.metric("Valor de Z na solução atual", format_km(metrics["total_km"]))
    st.write(
        "O algoritmo exato usa programação dinâmica Held-Karp. "
        "A heurística usa vizinho mais próximo seguido de melhoria local 2-opt."
    )
