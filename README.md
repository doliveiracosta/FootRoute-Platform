# RotaRecife

Aplicacao em Streamlit para simular rotas de entregadores de pedidos na Regiao Metropolitana do Recife.

**Desenvolvedor:** David de Oliveira Costa, Doutorando em Engenharia de Computacao, 2026.

- Perfil academico: <https://orcid.org/0000-0002-6138-7451>
- Perfil profissional: <https://www.linkedin.com/in/daviddeoliveiracosta/>

O painel permite selecionar um bairro de partida, escolher pedidos simulados, calcular uma rota recomendada e visualizar o grafo diretamente sobre o mapa.

## Escopo

- Area de aplicacao: Regiao Metropolitana do Recife-PE.
- Problema operacional: ordenacao de entregas para reduzir deslocamento, tempo e custo.
- Modelo-base: problema do caixeiro viajante para um entregador.
- Evolucao natural: problema de roteamento de veiculos quando houver varios entregadores, janelas de tempo ou capacidade.

## Funcionalidades

- Selecao do bairro de partida.
- Selecao dos pedidos a entregar.
- Retorno opcional ao ponto de partida.
- Algoritmo exato Held-Karp.
- Mapa com grafo esquematico da rota.
- Tabela de sequencia e trechos.
- Estimativa aproximada de distancia, tempo e custo operacional.
- Exportacao dos trechos em CSV.

## Publicacao no Streamlit Cloud

Use:

```text
Main file path: app.py
```

Em **Advanced settings**, selecione:

```text
Python version: 3.12
```

Essa configuracao evita instabilidades observadas com Python 3.14 no Streamlit Cloud. Se o log continuar mostrando `Using Python 3.14.6 environment`, edite as configuracoes do app no Streamlit Cloud ou recrie o app selecionando Python 3.12 antes do deploy.

## Execucao local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Observacao

As distancias sao estimadas a partir de coordenadas geograficas e de um fator de ajuste viario. O tracado no mapa liga os pontos por segmentos retos e deve ser interpretado como representacao esquematica da ordem de visita, nao como percurso real pelas ruas. Para uso operacional real, a aplicacao pode ser integrada futuramente a dados de malha viaria, APIs de roteamento e informacoes de transito.
