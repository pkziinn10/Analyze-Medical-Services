# Conclusão do cenário experimental final

## Estado da evidência

Fonte: `src/src/artifacts/revisao_experimental/20260914T000127Z/`.

O experimento persistido usou `n=714`, target `0 = 0–1 consulta` e `1 = 2+ consultas`, nested CV estratificada 10×5 e seed 42. A execução gravou 135 artefatos, incluindo OOF, curvas ROC/PR, seleções de ensemble, limiares, pesos, MDI, Wilcoxon/Holm e manifesto.

## Resultados disponíveis

| Cenário/modelo | Macro F1 | F1 classe 0 | Recall classe 0 | Balanced accuracy | MCC | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Dummy | 0,4495 | 0,0000 | 0,0000 | 0,5000 | 0,0000 | 0,5000 |
| Baseline NB | 0,4930 | 0,1979 | 0,2357 | 0,5030 | 0,0092 | 0,5522 |
| Baseline NC | 0,4582 | 0,2894 | 0,5407 | 0,5225 | 0,0352 | 0,5420 |
| MDI 80% MLP | 0,4977 | 0,1133 | 0,0758 | 0,5147 | 0,0692 | 0,5196 |
| MDI 80% NC | 0,4713 | 0,2870 | 0,5033 | 0,5252 | 0,0429 | 0,5330 |
| Custo sensível LR | 0,4769 | 0,3168 | 0,5956 | 0,5533 | 0,0824 | 0,5743 |
| Custo sensível RF | 0,4769 | 0,2446 | 0,3918 | 0,5149 | 0,0218 | 0,5407 |

Médias dos 10 folds externos, extraídas de `metrics_per_fold.csv`.

## Interpretação prudente

- A acurácia associada à classe majoritária não é evidência suficiente de discriminação.
- Dummy, AdaBoost, LR e SVM baseline tiveram F1 e recall da classe 0 nulos; portanto, não identificaram adequadamente o grupo minoritário.
- Custo sensível em LR elevou recall da classe 0 para 0,5956 e balanced accuracy para 0,5533, mas macro F1 permaneceu 0,4769 e MCC 0,0824. O ganho é limitado.
- MDI 80% não demonstrou ganho robusto: o melhor macro F1 disponível foi 0,4977, próximo ao comportamento de baixa discriminação observado nos demais cenários.
- Resultados de Wilcoxon com Holm persistidos não indicaram contraste estatisticamente significativo. Com 10 folds dependentes, esses testes são exploratórios.

## Observações da banca: situação técnica

| Observação | Situação no código/artefatos |
|---|---|
| Target, distribuição e classe positiva explícitos | Implementado |
| Nested CV 10×5 e busca manual | Implementado |
| AUROC/AP por scores, não `predict()` | Implementado para modelos com score; OPF sem score fica indisponível |
| Dummy, métricas por classe, balanced accuracy, MCC, OOF e matrizes | Implementados |
| MDI interno 80% e custo interno | Implementados |
| Ensemble e limiar internos | Implementados e exportados em OOF, seleções de membros, scores e limiares |
| Wilcoxon pareado com Holm | Implementado; resultado disponível é exploratório e não significativo |
| Artefatos reproduzíveis finais | Exportados; 135 arquivos no diretório do experimento |

## Limitações e conclusão

Estudo com 714 participantes, desenho transversal, única base e ausência de validação externa. O desbalanceamento permanece central. Importância preditiva não implica causalidade, utilidade clínica, triagem ou diagnóstico.

Os resultados não sustentam alegação de desempenho clínico, prontidão para uso ou superioridade robusta. A conclusão defensável é que custo sensível em regressão logística mostrou melhora limitada de detecção da classe minoritária, ainda com discriminação modesta. Atualizações do artigo devem usar exclusivamente os artefatos corrigidos e manter essa interpretação prudente.
