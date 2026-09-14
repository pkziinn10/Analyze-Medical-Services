# Metodologia do experimento de ML

Notebook metodológico: `src/AUSMI.ipynb`.

Este documento explica somente como o experimento é construído e validado. Resultados numéricos, interpretação científica e texto do artigo ficam fora deste guia.

## 1. Pergunta experimental

O experimento avalia se as variáveis disponíveis permitem classificar utilização de serviços médicos em dois grupos. Não estima causalidade, não diagnostica indivíduos e não recomenda intervenção clínica.

## 2. Dados e target

Entrada: `src/NPHA-doctor-visits.csv`.

- 714 participantes.
- Target cru: `Number of Doctors Visited`.
- Mapeamento fixo: código cru `1` vira classe `0`; códigos `2` e `3` viram classe `1`.
- Classe final `0`: 0–1 consulta; minoritária.
- Classe final `1`: 2+ consultas; positiva e majoritária.

Antes do treino, o notebook valida tamanho da base, presença do target, códigos permitidos, contagens por classe, percentuais e hash do CSV. Se algo divergir, a execução deve parar.

## 3. Desenho do pipeline

```text
CSV bruto
   |
   +--> validar target, contagens, hash e seed
   |
   +--> X bruto + y binário
           |
           +--> CV externa estratificada: 10 folds
                    |
                    +--> treino externo ------------------------------+
                    |                                                  |
                    |     CV interna estratificada: 5 folds            |
                    |          |                                       |
                    |          +--> preprocessar apenas treino interno |
                    |          +--> MDI/pesos/calibração internos      |
                    |          +--> ParameterGrid e macro F1           |
                    |                                                  |
                    +--> reajustar vencedor no treino externo ---------+
                    |
                    +--> teste externo uma única vez
                              |
                              +--> métricas, scores e OOF
                                      |
                                      +--> artefatos CSV/JSON/curvas
```

O princípio central é simples: qualquer escolha adaptativa ocorre antes de acessar o teste externo daquele fold.

## 4. Nested cross-validation

### Camada externa

`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` divide a base em 10 testes externos. Estratificação preserva aproximadamente proporção das classes em cada fold.

Cada participante aparece uma vez no teste externo. Ao fim, as predições dos 10 testes formam OOF com 714 registros.

### Camada interna

Dentro de cada treino externo, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` seleciona parâmetros. Para cada combinação do `ParameterGrid`, o critério é macro F1 médio dos cinco folds internos.

O modelo vencedor é treinado novamente usando o treino externo completo. Só depois ele prevê teste externo.

## 5. Preprocessamento sem leakage

O helper `prepare_split` cria objetos novos por ajuste:

1. identifica colunas numéricas e categóricas;
2. imputa numéricas pela mediana do treino;
3. imputa categóricas pela moda do treino;
4. aplica one-hot encoding com categorias vistas somente no treino;
5. aplica escala nas numéricas quando necessária;
6. transforma validação/teste usando objetos já ajustados.

Não é permitido ajustar imputer, encoder, scaler ou seletor em toda base antes dos folds. Mesmo uma transformação aparentemente simples pode carregar informação da validação/teste.

## 6. Busca de hiperparâmetros

Todos os grids são declarados no notebook. `ParameterGrid` percorre combinações manualmente. Para cada combinação:

1. cria preprocessador novo;
2. ajusta no treino interno;
3. prevê validação interna;
4. registra macro F1 e acurácia;
5. calcula média pelos cinco folds internos.

O maior macro F1 decide configuração. Em empates, a implementação deve manter critério determinístico. Nunca reutilizar objeto da última combinação testada como modelo final.

## 7. Métricas e scores

A matriz de confusão sempre usa `labels=[0, 1]`:

```text
[[TN, FP],
 [FN, TP]]
```

Por fold, registrar:

- precision, recall e F1 das classes 0 e 1;
- macro/weighted F1;
- macro precision e macro recall;
- balanced accuracy e MCC;
- specificity e sensitivity;
- AUROC e AP/PR-AUC;
- matriz de confusão;
- classe prevista e score OOF.

AUROC e AP usam `predict_proba` ou `decision_function`, nunca `predict`. Se um modelo não fornece score contínuo, AUROC/AP ficam indisponíveis (`NaN`) e não se gera curva ROC/PR para ele.

## 8. Fases do experimento

### Fase 1 — Baseline

Compara Dummy e modelos individuais usando todas as features. Dummy define referência mínima: se um modelo não melhora métricas relevantes da classe minoritária além do Dummy, não há evidência de ganho preditivo útil.

Não há seleção MDI, peso de classe, ensemble ou limiar ajustado nesta fase.

### Fase 2 — MDI 80%

MDI é calculada dentro do treino de cada split. Uma RF auxiliar mede importância em matriz já transformada. As importâncias são ordenadas e o notebook retém o menor prefixo cuja soma acumulada é pelo menos 80%.

O seletor é ajustado novamente em cada treino externo antes do ajuste final. Exportar: importâncias, acumulado, atributos selecionados e frequência de seleção.

MDI é mecanismo preditivo; não deve ser interpretada como efeito causal ou relevância clínica.

### Fase 3 — Custo sensível

Pesos de classe são calculados a partir do vetor `y_fit` usado naquele ajuste. Logo, pesos de um fold interno usam apenas treino interno; pesos do refit usam treino externo completo.

Somente modelos compatíveis recebem custo. Modelos sem mecanismo de peso ficam marcados como não aplicáveis, em vez de receber tratamento artificial.

### Fase 4 — Ensemble

O ensemble começa com candidatos capazes de fornecer score calibrável. Para cada fold externo:

1. cada candidato gera perfil OOF interno;
2. perfil inclui macro F1, métricas da classe 0, balanced accuracy, MCC, AP e vetor de erros;
3. primeiro membro é escolhido por qualidade interna;
4. segundo e terceiro combinam qualidade e diversidade real de erros;
5. três membros únicos são ajustados no treino externo;
6. teste externo recebe voto majoritário e voto probabilístico calibrado.

Voto majoritário usa classes dos três membros. Voto probabilístico usa média ponderada de probabilidades calibradas; pesos somam 1. Membros não podem ser selecionados pelo teste externo.

### Fase 5 — Limiar

O limiar padrão é 0,5. O limiar alternativo é escolhido somente em OOF interno do ensemble probabilístico, usando critério pré-definido de macro F1, com desempates por balanced accuracy, MCC e proximidade de 0,5.

Os dois limiares são aplicados ao mesmo score externo. Não há novo ajuste de modelo depois da escolha de limiar.

## 9. Estatística

Wilcoxon compara métricas dos mesmos 10 folds. Holm ajusta múltiplas comparações:

```text
ordenar p brutos
aplicar multiplicadores decrescentes
garantir máximo cumulativo
limitar resultado ao intervalo [0, 1]
restaurar ordem dos contrastes
```

Com 10 folds, esse resultado é exploratório. Deve registrar estatística, p bruto, p Holm, diferença média, número de folds e limitação de dependência entre folds.

## 10. Reprodutibilidade e verificações

Cada execução deve criar um `run_id` e salvar manifesto com seed, hash do dado, versões e configurações. Antes de aceitar uma execução, verificar:

1. 10 folds externos disjuntos;
2. cada índice aparece uma vez em OOF externo;
3. OOF tem 714 registros em cada cenário/modelo aplicável;
4. modelo com score válido tem scores finitos em todos folds;
5. modelo sem score tem indisponibilidade explícita;
6. MDI alcança ao menos 80% acumulado;
7. ensemble contém três membros diferentes;
8. pesos probabilísticos somam 1;
9. limiar foi escolhido por OOF interno;
10. valores Holm ficam entre 0 e 1.

## 11. Artefatos metodológicos

| Artefato | O que permite auditar |
|---|---|
| `manifest.json` | dados, seed, versões, limitações e inventário |
| `folds.csv` | separação externa treino/teste |
| `search.csv` | decisão interna de hiperparâmetros |
| `metrics_per_fold.csv` | resultado por fold sem esconder variabilidade |
| `oof_*.csv` | cobertura OOF, classes e scores |
| `mdi_detailed.json`, `mdi_frequency.csv` | seleção de atributos MDI |
| `weights.json` | pesos e dados de treino usados |
| `ensemble_profiles.json`, `ensemble_members.json` | seleção e diversidade do ensemble |
| `thresholds.json` | limiares internos por fold |
| `wilcoxon_holm.json` | pares e ajuste estatístico |
| `roc_*`, `pr_*` | curvas apenas de scores válidos |

## 12. Regras que não podem ser quebradas

- Não usar SMOTE; não gerar registros sintéticos nesta base sensível.
- Não usar teste externo para escolher parâmetro, atributo, peso, membro ou limiar.
- Não comunicar acurácia isolada como desempenho em base desbalanceada.
- Não substituir score contínuo por classe prevista em AUROC/AP.
- Não transformar importância em causalidade.
- Não misturar outputs de runs diferentes.
