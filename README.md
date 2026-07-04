# FootRoute

FootRoute é uma plataforma em Streamlit para modelar, otimizar e visualizar rotas logísticas entre equipes de futebol. O painel usa uma formulação inspirada no Problema do Caixeiro Viajante, mas acrescenta uma camada interativa que conecta escolhas do usuário, grafo, variáveis de decisão, função objetivo e restrições matemáticas.

## O que esta versão contém

- seleção do clube de origem;
- seleção das equipes a visitar;
- opção de retornar ou não à origem;
- algoritmo exato Held-Karp para instâncias pequenas;
- heurística vizinho mais próximo + 2-opt para instâncias maiores;
- grafo da rota recomendada;
- tabela dos trechos da rota;
- matriz de distâncias entre clubes;
- download da rota em CSV;
- aba de modelagem matemática interativa com equações renderizadas por `st.latex()`;
- variáveis de decisão ativadas na solução;
- função objetivo numérica formada pelos trechos selecionados;
- classificação territorial das equipes: capital com múltiplas equipes, capital com uma equipe e equipes do interior.

## Base de equipes incluída

### Capitais com múltiplas equipes

- São Paulo/SP: Corinthians, Palmeiras e São Paulo;
- Rio de Janeiro/RJ: Botafogo, Flamengo, Fluminense e Vasco da Gama;
- Belo Horizonte/MG: Atlético-MG e Cruzeiro;
- Curitiba/PR: Athletico-PR e Coritiba;
- Porto Alegre/RS: Grêmio e Internacional;
- Salvador/BA: Bahia e Vitória.

### Capital com apenas uma equipe

- Belém/PA: Remo.

### Equipes do interior

- Bragança Paulista/SP: Red Bull Bragantino;
- Chapecó/SC: Chapecoense;
- Mirassol/SP: Mirassol;
- Santos/SP: Santos.

## Interpretação territorial

A plataforma separa três entidades:

1. **Clube**: a entidade esportiva.
2. **Cidade-sede**: a cidade real onde a equipe está localizada.
3. **Capital de referência**: capital estadual usada para análise agregada.

Exemplo:

```text
Santos = clube
Santos/SP = cidade-sede
São Paulo/SP = capital de referência
```

Isso evita tratar equipes do interior como se estivessem fisicamente localizadas na capital.

## Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Depois, acesse:

```text
http://localhost:8501
```

## Estrutura

```text
footroute-platform/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── clubes.csv
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

## Observação sobre distâncias

As distâncias são estimadas pela fórmula de Haversine a partir das coordenadas geográficas das cidades-sede. Portanto, representam distância geodésica aproximada, não distância rodoviária, aérea operacional ou custo logístico real.
