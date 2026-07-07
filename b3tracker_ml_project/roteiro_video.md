# Roteiro para vídeo de demonstração: até 5 minutos

## 0:00-0:40 — Contexto

Apresentar o B3Tracker original:

> O B3Tracker é uma aplicação em Django que monitora cotações da B3 e envia alertas por e-mail quando o preço cruza limites manuais. O problema é que limites fixos não acompanham volatilidade nem tendência de curto prazo.

Explicar o objetivo:

> Neste projeto, eu adicionei uma camada de Machine Learning para classificar cada ativo como COMPRA, VENDA ou NEUTRO.

## 0:40-1:30 — Estrutura do repositório

Mostrar rapidamente:

```bash
ls
ls src/b3tracker_ml
```

Comentar:

> O projeto está separado em coleta de dados, engenharia de features, criação do alvo, treinamento, avaliação e inferência de alertas.

## 1:30-2:20 — Execução do pipeline

Rodar:

```bash
python scripts/run_pipeline.py --source yfinance --model random_forest
```

Se estiver sem internet:

```bash
python scripts/run_pipeline.py --source demo --model random_forest
```

Explicar:

> O modo `yfinance` baixa dados reais. O modo `demo` existe apenas para demonstrar o pipeline quando não há internet.

## 2:20-3:20 — Dataset e features

Mostrar:

```bash
head data/processed/dataset_modelagem.csv
cat data/processed/class_distribution.csv
```

Explicar:

> Cada linha representa um ativo em uma data. As features incluem retornos, médias móveis, RSI, MACD, volatilidade, Bollinger, volume e contexto de mercado com BOVA11.

## 3:20-4:20 — Avaliação

Mostrar:

```bash
cat data/processed/test_risk_threshold_metrics.json
cat data/processed/test_risk_threshold_confusion_matrix.csv
```

Explicar:

> A métrica mais importante não é só acurácia. Eu olho a matriz de confusão porque falso positivo de compra é mais perigoso do que falso negativo. Por isso apliquei um limiar conservador de probabilidade.

## 4:20-5:00 — Alertas finais e limitações

Mostrar:

```bash
cat data/processed/latest_alerts.csv
```

Fechar com:

> O modelo não substitui análise financeira e não usa notícias ou fundamentos. Ele funciona como uma camada de apoio para tornar os alertas do B3Tracker menos estáticos e mais sensíveis ao regime recente do mercado.
