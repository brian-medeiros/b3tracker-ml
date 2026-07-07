# B3Tracker ML: Alertas Inteligentes para Ativos da B3

Este repositório implementa uma camada de Aprendizado de Máquina para evoluir o projeto B3Tracker. A aplicação original monitora ativos financeiros da B3 e dispara alertas por e-mail quando o preço ultrapassa limites manuais definidos pelo usuário. A proposta deste projeto é adicionar uma camada preditiva capaz de classificar janelas de curto prazo como `COMPRA`, `VENDA` ou `NEUTRO`, usando indicadores técnicos calculados a partir de cotações históricas.

O objetivo não é criar um robô de investimento, mas sim demonstrar uma solução de engenharia que melhora o mecanismo de alertas: em vez de depender apenas de túneis fixos de preço, o sistema passa a considerar volatilidade, tendência, volume e contexto de mercado.

## Estrutura do projeto

```text
b3tracker_ml_project/
├── data/
│   ├── raw/                  # Cotações baixadas ou arquivo CSV de entrada
│   └── processed/            # Dataset final, métricas e matrizes de confusão
├── integration/              # Exemplo de integração com Django/B3Tracker
├── outputs/
│   ├── figures/              # Figuras geradas pelo pipeline
│   └── models/               # Modelo treinado
├── reports/
│   ├── relatorio_tecnico.md                  # Fonte editável do relatório
│   ├── relatorio_tecnico_b3tracker_ml.pdf    # Relatório final para entrega
│   └── roteiro_video_b3tracker_ml.pdf        # Roteiro em PDF
├── scripts/
│   └── run_pipeline.py       # Script principal de reprodução
├── src/
│   └── b3tracker_ml/         # Código modular do projeto
├── README.md
├── requirements.txt
└── roteiro_video.md
```

## Instalação

Use Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução recomendada com dados reais

O comando abaixo baixa cotações via `yfinance`, calcula as features, treina um Random Forest e salva métricas, figuras e modelo:

```bash
python scripts/run_pipeline.py --source yfinance --model random_forest
```

Para executar com XGBoost:

```bash
python scripts/run_pipeline.py --source yfinance --model xgboost
```

## Execução offline para demonstração

Caso a máquina esteja sem acesso à internet, é possível rodar o pipeline com uma base sintética determinística. Essa opção serve apenas para demonstrar o funcionamento do código, não para sustentar conclusões financeiras.

```bash
python scripts/run_pipeline.py --source demo --model random_forest
```

## Execução usando CSV próprio

O CSV deve conter as colunas:

```text
ticker,date,open,high,low,close,adjusted_close,volume
```

Comando:

```bash
python scripts/run_pipeline.py --source csv --csv-path data/raw/minhas_cotacoes.csv --model random_forest
```

## Arquivos gerados

Após a execução, os principais arquivos são:

| Arquivo | Descrição |
|---|---|
| `data/processed/dataset_modelagem.csv` | Dataset tabular final usado no treino |
| `data/processed/class_distribution.csv` | Distribuição das classes |
| `data/processed/test_raw_metrics.json` | Métricas sem limiar conservador |
| `data/processed/test_risk_threshold_metrics.json` | Métricas com limiar conservador |
| `data/processed/test_risk_threshold_confusion_matrix.csv` | Matriz de confusão principal |
| `data/processed/threshold_sweep_validation.csv` | Teste de diferentes limiares de decisão |
| `data/processed/feature_importance.csv` | Importância das features |
| `data/processed/latest_alerts.csv` | Alertas mais recentes por ativo |
| `outputs/models/random_forest_b3tracker.joblib` | Modelo treinado |

Se `matplotlib` e `seaborn` estiverem instalados, também serão salvas figuras em `outputs/figures/`.

## Formulação do problema

Cada linha do dataset representa um ativo em uma data. As variáveis explicativas são calculadas apenas com informações disponíveis até aquela data, evitando vazamento de dados. O rótulo é definido pelo retorno futuro em 5 dias úteis.

Regra principal:

```text
COMPRA: retorno futuro >= limite dinamico
VENDA:  retorno futuro <= -limite dinamico
NEUTRO: caso contrário
```

O limite dinâmico considera a volatilidade recente:

```text
limite dinamico = max(2%, volatilidade_20d * sqrt(5))
```

## Features utilizadas

As principais famílias de features são:

| Grupo | Exemplos |
|---|---|
| Retornos | `retorno_1d`, `retorno_5d`, `retorno_20d` |
| Tendência | distância para médias móveis de 5, 20, 50 e 200 dias |
| Momentum | RSI de 7 e 14 dias, MACD |
| Volatilidade | volatilidade diária e anualizada, razão `vol_5d / vol_20d` |
| Bollinger | posição nas bandas e largura das bandas |
| Volume | volume relativo, z-score de volume |
| Mercado | retorno do BOVA11, volatilidade do índice, beta de 60 dias |
| Calendário | dia da semana, mês, início/fim de mês |

## Avaliação

A avaliação respeita a ordem temporal dos dados. O projeto não usa split aleatório simples, pois isso misturaria passado e futuro.

Além da acurácia balanceada, o foco está na matriz de confusão:

| Erro | Impacto |
|---|---|
| Falso positivo de compra | Pode induzir uma compra ruim |
| Falso negativo de compra | Perde uma oportunidade |
| Compra prevista como venda | Erro direcional grave |
| Neutro previsto como compra/venda | Gera alerta desnecessário |

Por isso, o projeto aplica um limiar conservador:

```text
emitir COMPRA somente se P(COMPRA) >= 0.60 e P(COMPRA) - P(VENDA) >= 0.15
emitir VENDA somente se P(VENDA) >= 0.60 e P(VENDA) - P(COMPRA) >= 0.15
caso contrário, manter NEUTRO
```

## Limitações

O modelo usa apenas dados técnicos históricos. Ele não incorpora notícias, resultados trimestrais, decisões de juros, eventos políticos, dividendos futuros ou choques externos. Portanto, os alertas devem ser tratados como apoio à decisão, não como recomendação automática de investimento.

## Demonstração em vídeo

O arquivo `roteiro_video.md` e sua versão em PDF em `reports/roteiro_video_b3tracker_ml.pdf` contêm uma sugestão de vídeo de até 5 minutos mostrando:

1. problema original do B3Tracker;
2. execução do pipeline;
3. dataset gerado;
4. matriz de confusão;
5. exemplos de alertas finais;
6. limitações do modelo.
