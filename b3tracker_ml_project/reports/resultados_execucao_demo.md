# Resultados da execução de demonstração

Este arquivo registra a execução feita no modo `demo`, que usa uma série sintética determinística. O objetivo é comprovar que o pipeline roda de ponta a ponta mesmo sem internet. Estes números não devem ser interpretados como desempenho financeiro real.

Comando executado:

```bash
python scripts/run_pipeline.py --source demo --model random_forest
```

Resumo da execução:

| Item | Valor |
|---|---:|
| Observações após tratamento | 6.164 |
| Features utilizadas | 58 |
| Linhas de treino | 4.329 |
| Linhas de validação | 899 |
| Linhas de teste | 936 |

Métricas no teste após limiar conservador:

| Métrica | Valor |
|---|---:|
| Balanced accuracy | 0,364 |
| Precision COMPRA | 0,167 |
| Recall COMPRA | 0,009 |
| Precision VENDA | 0,714 |
| Recall VENDA | 0,116 |
| Macro F1 | 0,311 |

Matriz de confusão após limiar conservador:

| Classe real \\ prevista | NEUTRO | COMPRA | VENDA |
|---|---:|---:|---:|
| NEUTRO | 502 | 5 | 12 |
| COMPRA | 111 | 1 | 2 |
| VENDA | 268 | 0 | 35 |

Interpretação:

O limiar conservador reduziu a emissão de sinais de compra e venda. Isso é coerente com a estratégia de risco do projeto, pois um falso positivo de compra é mais grave do que perder uma oportunidade. Entretanto, o recall de `COMPRA` ficou muito baixo na execução de demonstração, indicando que, em dados reais, o limiar deve ser calibrado usando a base de validação e a matriz de confusão.

Principais features por importância no modo demo:

| Feature | Importância |
|---|---:|
| `dist_mm_50` | 0,121 |
| `macd` | 0,105 |
| `dist_mm20_mm50` | 0,097 |
| `macd_signal` | 0,058 |
| `bollinger_position` | 0,056 |

Conclusão da execução:

A execução confirma que o projeto gera dataset, treina modelo, calcula matriz de confusão, aplica limiar de risco e produz alertas finais. Para a entrega acadêmica, a execução principal deve ser feita com `--source yfinance` para usar dados reais da B3.
