# FootRoute

FootRoute é uma plataforma em Streamlit para estimar e visualizar rotas logísticas entre clubes de futebol. O painel usa uma formulação inspirada no problema do caixeiro viajante para sugerir o percurso de menor distância a partir de um clube de origem.

## Funcionalidades

- seleção do clube de origem;
- seleção dos clubes a visitar;
- opção de retorno ao clube de origem;
- algoritmo exato Held-Karp;
- heurística vizinho mais próximo + 2-opt;
- grafo da rota recomendada;
- tabela de trechos com distância, região e indicação de viagem longa;
- matriz de distâncias entre clubes;
- comparação com a ordem inicial dos clubes;
- download da rota em CSV.

## Dados incluídos

O protótipo usa um recorte de 13 clubes:

- Atlético-MG;
- Bahia;
- Botafogo;
- Corinthians;
- Cruzeiro;
- Flamengo;
- Fluminense;
- Grêmio;
- Internacional;
- Palmeiras;
- Santos;
- São Paulo;
- Vasco.

Também inclui as 27 capitais brasileiras como pontos referenciais.

As distâncias são estimadas pela fórmula de Haversine a partir das coordenadas geográficas das cidades-sede. Portanto, representam uma aproximação geodésica, não uma distância rodoviária ou aérea operacional real.

## Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Depois, abra:

```text
http://localhost:8501
```

## Como publicar no GitHub

1. Crie um repositório no GitHub.
2. Envie todos os arquivos deste diretório.
3. Confirme que `requirements.txt`, `app.py`, `data/` e `src/` estão na raiz do repositório.
4. Para publicar no Streamlit Community Cloud, selecione o repositório e defina:

```text
Main file path: app.py
```

## Estrutura

```text
footroute-platform/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── clubes_13.csv
│   └── capitais_brasil.csv
├── docs/
│   └── modelo_matematico.md
└── src/
    └── footroute/
        ├── __init__.py
        ├── models.py
        ├── optimization.py
        └── visualization.py
```

## Possíveis extensões

- substituir distâncias geodésicas por distância aérea real ou tempo de viagem;
- inserir custo monetário estimado por trecho;
- considerar emissões de CO2;
- comparar rota oficial, rota otimizada e rota heurística;
- incluir restrições de calendário, descanso e sequência de mandos.
