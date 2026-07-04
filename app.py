from __future__ import annotations

from pathlib import Path
import sys
from math import atan2

import pandas as pd
import pydeck as pdk
import streamlit as st


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


def build_route_map(clubs: list[Place], route: list[Place], total_distance_label: str) -> pdk.Deck:
    route_names = {p.name for p in route}

    all_points = pd.DataFrame(
        {
            "name": [p.name for p in clubs],
            "city": [getattr(p, "city_label", f"{p.city}/{p.state}") for p in clubs],
            "lat": [p.lat for p in clubs],
            "lon": [p.lon for p in clubs],
            "in_route": [p.name in route_names for p in clubs],
        }
    )

    route_points = all_points[all_points["in_route"]].copy()
    other_points = all_points[~all_points["in_route"]].copy()

    path_df = pd.DataFrame(
        {
            "path": [[[p.lon, p.lat] for p in route]],
        }
    )

    # marcador de quilometragem dentro do mapa
    label_df = pd.DataFrame(
        [
            {
                "lon": -39.5,
                "lat": 3.0,
                "text": f"Distância total: {total_distance_label}",
            }
        ]
    )

    # círculos azuis numerados ao longo da rota, distribuídos com leve deslocamento lateral
    step_records: list[dict[str, float | int | str]] = []
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        lon1, lat1 = a.lon, a.lat
        lon2, lat2 = b.lon, b.lat
        dx, dy = lon2 - lon1, lat2 - lat1
        angle = atan2(dy, dx)
        nx, ny = -dy, dx
        norm = max((nx * nx + ny * ny) ** 0.5, 1e-9)
        nx /= norm
        ny /= norm
        t = 0.42 if i % 2 == 0 else 0.58
        offset = 0.45 if i % 2 == 0 else -0.45
        mlon = lon1 + dx * t + nx * offset
        mlat = lat1 + dy * t + ny * offset
        step_records.append({"lon": mlon, "lat": mlat, "step": str(i + 1)})

    steps_df = pd.DataFrame(step_records)

    layers = [
        pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_color=[59, 130, 246],
            width_scale=1,
            width_min_pixels=3,
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=other_points,
            get_position='[lon, lat]',
            get_radius=45000,
            get_fill_color=[148, 163, 184, 160],
            get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=route_points,
            get_position='[lon, lat]',
            get_radius=62000,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[239, 68, 68, 255],
            line_width_min_pixels=3,
            stroked=True,
            filled=False,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=steps_df,
            get_position='[lon, lat]',
            get_radius=82000,
            get_fill_color=[37, 99, 235, 230],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            data=steps_df,
            get_position='[lon, lat]',
            get_text='step',
            get_size=14,
            size_units='pixels',
            get_color=[255, 255, 255, 255],
            get_alignment_baseline='center',
            get_text_anchor='middle',
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            data=label_df,
            get_position='[lon, lat]',
            get_text='text',
            get_size=18,
            size_units='pixels',
            get_color=[15, 23, 42, 255],
            get_alignment_baseline='top',
            get_text_anchor='start',
            pickable=False,
        ),
    ]

    view_state = pdk.ViewState(latitude=-15.5, longitude=-52.5, zoom=3.25, pitch=0)

    tooltip = {
        "html": "<b>{name}</b><br/>{city}",
        "style": {"backgroundColor": "white", "color": "#111827"},
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip=tooltip,
    )


st.set_page_config(page_title="FootRoute", layout="wide")

clubs, capitals = load_data()
clubs_by_name = {club.name: club for club in clubs}
club_names = list(clubs_by_name)

st.title("FootRoute")
st.caption("Painel de otimização de rotas logísticas entre clubes de futebol.")
st.caption("VERSÃO ATIVA: rota em mapa do Brasil; apenas distância total; sem referencial e sem leitura operacional.")

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
    st.pydeck_chart(build_route_map(clubs, route, total_distance_label), use_container_width=True, height=700)
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
