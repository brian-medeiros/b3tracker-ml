# Como entregar o projeto

1. Crie um repositório público no GitHub.
2. Envie todos os arquivos desta pasta para o repositório.
3. Rode o pipeline principal:

```bash
python scripts/run_pipeline.py --source yfinance --model random_forest
```

4. Confira os arquivos gerados em `data/processed/` e `outputs/figures/`.
5. Atualize o relatório técnico com as métricas da sua execução real, se desejar.
6. Grave o vídeo seguindo `roteiro_video.md`.
7. Entregue:

| Material | Onde está |
|---|---|
| Código | Repositório GitHub |
| README | `README.md` |
| Relatório final | `reports/relatorio_tecnico_b3tracker_ml.pdf` |
| Fonte editável do relatório | `reports/relatorio_tecnico.md` |
| Roteiro do vídeo | `reports/roteiro_video_b3tracker_ml.pdf` e `roteiro_video.md` |
| Resultados reproduzíveis | `data/processed/` após execução |

Observação: o modo `demo` serve apenas para mostrar funcionamento sem internet. Para a versão final, prefira executar com dados reais via `yfinance`.
