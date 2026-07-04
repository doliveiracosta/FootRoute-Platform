from __future__ import annotations

from pathlib import Path
import sys
import html

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


def brazil_map_svg(clubs: list[Place], route: list[Place], total_distance_label: str) -> str:
    """Mapa esquemático do Brasil, sem mapa-múndi/base externa."""
    width, height = 1120, 760
    min_lon, max_lon = -74.5, -33.0
    min_lat, max_lat = -34.5, 6.0
    pad_l, pad_r, pad_t, pad_b = 70, 70, 95, 45

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = pad_l + (lon - min_lon) / (max_lon - min_lon) * (width - pad_l - pad_r)
        y = pad_t + (max_lat - lat) / (max_lat - min_lat) * (height - pad_t - pad_b)
        return x, y

    # Contorno aproximado, suficiente para usar como fundo visual do Brasil sem depender de tiles externos.
    outline_lonlat = [
        (-73.9, -7.5), (-72.2, -3.4), (-69.8, 0.5), (-64.4, 2.9),
        (-59.8, 5.2), (-54.6, 4.9), (-50.4, 2.4), (-47.0, 1.0),
        (-44.7, -1.3), (-41.0, -2.6), (-37.0, -5.0), (-34.8, -7.7),
        (-35.2, -11.0), (-38.0, -13.2), (-38.8, -17.2), (-40.6, -19.8),
        (-43.2, -22.7), (-45.8, -23.7), (-48.4, -25.5), (-48.8, -28.4),
        (-51.1, -31.3), (-53.3, -33.7), (-56.5, -30.9), (-57.6, -28.7),
        (-54.2, -25.6), (-54.8, -22.5), (-58.6, -20.7), (-60.8, -17.4),
        (-63.6, -13.4), (-67.6, -10.8), (-70.2, -9.2), (-73.9, -7.5),
    ]

    outline_points = " ".join(f"{project(lon, lat)[0]:.1f},{project(lon, lat)[1]:.1f}" for lon, lat in outline_lonlat)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" '
        'style="background:#f8fafc;border-radius:14px;font-family:Arial, Helvetica, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#f8fafc"/>',
        '<text x="26" y="38" font-size="21" font-weight="700" fill="#0f172a">Rota sobre o mapa do Brasil</text>',
        '<text x="26" y="64" font-size="13" fill="#334155">Mapa esquemático do Brasil; pontos = clubes/cidades-sede; linhas = deslocamentos selecionados</text>',
        f'<polygon points="{outline_points}" fill="#e5e7eb" stroke="#94a3b8" stroke-width="2.2" opacity="0.96"/>',
        '<rect x="800" y="22" width="285" height="58" rx="10" fill="#ffffff" opacity="0.94" stroke="#cbd5e1" stroke-width="1"/>',
        '<text x="818" y="45" font-size="13" font-weight="700" fill="#334155">Distância total</text>',
        f'<text x="818" y="68" font-size="23" font-weight="700" fill="#0f172a">{html.escape(total_distance_label)}</text>',
    ]

    point_by_name = {place.name: project(place.lon, place.lat) for place in clubs}
    route_names = {place.name for place in route}

    # Linhas da rota.
    if len(route) >= 2:
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            x1, y1 = point_by_name[a.name]
            x2, y2 = point_by_name[b.name]
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#2563eb" stroke-width="3.4" stroke-linecap="round" opacity="0.92"/>'
            )

            dx, dy = x2 - x1, y2 - y1
            seg_len = max((dx * dx + dy * dy) ** 0.5, 1.0)
            nx, ny = -dy / seg_len, dx / seg_len
            t = 0.42 if i % 2 == 0 else 0.58
            offset = 13 if i % 2 == 0 else -13
            mx = x1 + dx * t + nx * offset
            my = y1 + dy * t + ny * offset
            svg.append(
                f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="9.5" fill="#2563eb" '
                'stroke="#ffffff" stroke-width="1.5" opacity="0.96"/>'
            )
            svg.append(
                f'<text x="{mx:.1f}" y="{my + 3.2:.1f}" text-anchor="middle" '
                'font-size="9.5" font-weight="700" fill="#ffffff">'
                f'{i + 1}</text>'
            )

    # Pontos. Clubes da rota ficam vazados em vermelho; demais pontos em cinza discreto.
    for place in clubs:
        x, y = point_by_name[place.name]
        if place.name in route_names:
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.8" fill="none" '
                'stroke="#ef4444" stroke-width="3.0"/>'
            )
        else:
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#94a3b8" '
                'stroke="#ffffff" stroke-width="1.2" opacity="0.75"/>'
            )

    svg.append("</svg>")
    return "".join(svg)


st.set_page_config(page_title="FootRoute", layout="wide")

clubs, capitals = load_data()
clubs_by_name = {club.name: club for club in clubs}
club_names = list(clubs_by_name)

st.title("FootRoute")
st.caption("Painel de otimização de rotas logísticas entre clubes de futebol.")
st.caption("VERSÃO ATIVA: mapa do Brasil apenas; sem mapa externo; sem aba Referenciais.")

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
_ = 0.0 if baseline_distance == 0 else (baseline_distance - total_distance) / baseline_distance * 100

total_distance_label = format_km(metrics["total_km"])

tab_route, tab_legs, tab_model = st.tabs(["Rota", "Trechos", "Modelo"])

with tab_route:
    components.html(brazil_map_svg(clubs, route, total_distance_label), height=780, scrolling=False)
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
    st.metric("Valor de Z na solução atual", total_distance_label)
    st.write(
        "O algoritmo exato usa programação dinâmica Held-Karp. "
        "A heurística usa vizinho mais próximo seguido de melhoria local 2-opt."
    )
