# Mapa do Exam Guide Oficial → Módulos deste Guia

> Cada linha abaixo é um **objetivo oficial** listado no [exam guide da Databricks](https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf) (versão de 1º de março de 2025), traduzido, com o link para onde ele é coberto neste repositório.
>
> Use esta página como **checklist final**: se você consegue explicar cada linha em voz alta, está pronto.

> ⚠️ **Material experimental e não oficial.** Este mapeamento é uma interpretação do autor sobre onde cada objetivo é coberto — não é endossado pela Databricks e não garante aprovação. **Sempre confira o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate) duas semanas antes da sua prova**, porque a Databricks atualiza o conteúdo sem aviso prévio. Veja o [aviso completo no README](README.md#️-aviso-importante--leia-antes-de-começar).

---

## Como ler esta página

- ✅ = coberto com teoria + código executável
- 📘 = coberto como conceito (não dá para praticar na Free Edition, ou é puramente conceitual na prova)

---

## Seção 1 — Databricks Machine Learning (38% da prova)

A maior seção. Concentra MLflow, Feature Store, Unity Catalog, AutoML e MLOps.

| | Objetivo oficial | Onde estudar |
|---|---|---|
| 📘 | Identificar as boas práticas de uma estratégia de MLOps | [11 — MLOps](modulos/11_mlops.md#111-o-que-é-mlops-na-prática) |
| 📘 | Identificar as vantagens de usar os ML runtimes | [01 — Plataforma](modulos/01_plataforma.md#13-databricks-runtime-e-runtime-ml) |
| 📘 | Identificar como o AutoML facilita a seleção de modelos/features | [08 — AutoML](modulos/08_automl.md#81-o-que-o-automl-faz-automaticamente) |
| 📘 | Identificar as vantagens que o AutoML traz ao desenvolvimento de modelos | [08 — AutoML](modulos/08_automl.md#82-as-vantagens-que-caem-na-prova) |
| 📘 | Identificar os benefícios de criar feature tables no nível de conta (Unity Catalog) vs no nível de workspace | [09 — Feature Store](modulos/09_feature_store.md#92-unity-catalog-vs-workspace-feature-store) |
| ✅ | Criar uma feature table no Unity Catalog | [09 — Feature Store](modulos/09_feature_store.md#93-criar-uma-feature-table-no-unity-catalog) |
| ✅ | Escrever dados em uma feature table | [09 — Feature Store](modulos/09_feature_store.md#94-escrever-e-ler-dados-da-feature-table) |
| ✅ | Treinar um modelo com features de uma feature table | [09 — Feature Store](modulos/09_feature_store.md#96-treinar-e-logar-o-modelo-vinculado-à-feature-store) |
| ✅ | Pontuar um modelo usando features de uma feature table | [09 — Feature Store](modulos/09_feature_store.md#97-scoring-em-batch-com-score_batch) |
| 📘 | Descrever as diferenças entre feature tables online e offline | [09 — Feature Store](modulos/09_feature_store.md#98-online-vs-offline-feature-tables) |
| ✅ | Identificar o melhor run usando a MLflow Client API | [04 — MLflow](modulos/04_mlflow.md#46-encontrar-o-melhor-run-com-a-mlflow-client-api) |
| ✅ | Logar manualmente métricas, artefatos e modelos em um MLflow Run | [04 — MLflow](modulos/04_mlflow.md#43-tracking--registrar-um-experimento-manualmente) |
| 📘 | Identificar as informações disponíveis na MLflow UI | [04 — MLflow](modulos/04_mlflow.md#45-o-que-você-vê-na-mlflow-ui) |
| ✅ | Registrar um modelo usando a MLflow Client API no registry do Unity Catalog | [04 — MLflow](modulos/04_mlflow.md#47-model-registry-no-unity-catalog) |
| 📘 | Identificar os benefícios de registrar modelos no Unity Catalog em vez do workspace registry | [04 — MLflow](modulos/04_mlflow.md#48-por-que-unity-catalog-e-não-o-workspace-registry) |
| 📘 | Identificar cenários em que promover código é preferível a promover modelos, e vice-versa | [11 — MLOps](modulos/11_mlops.md#113-promover-código-ou-promover-modelo) |
| ✅ | Definir ou remover uma tag de um modelo | [04 — MLflow](modulos/04_mlflow.md#49-tags-e-descrições) |
| ✅ | Promover um modelo challenger a champion usando aliases | [04 — MLflow](modulos/04_mlflow.md#410-aliases-champion-e-challenger) |

---

## Seção 2 — Data Processing (19% da prova)

A seção mais negligenciada por quem estuda só a parte de MLflow. São questões conceituais de estatística e preparação de dados.

| | Objetivo oficial | Onde estudar |
|---|---|---|
| ✅ | Calcular estatísticas descritivas de um DataFrame Spark com `.summary()` ou dbutils data summaries | [03 — EDA](modulos/03_eda_preparacao.md#31-estatísticas-descritivas) |
| ✅ | Remover outliers de um DataFrame Spark com base em desvio padrão ou IQR | [03 — EDA](modulos/03_eda_preparacao.md#32-detectar-e-remover-outliers) |
| ✅ | Criar visualizações para features categóricas ou contínuas | [03 — EDA](modulos/03_eda_preparacao.md#33-visualizações) |
| ✅ | Comparar duas features categóricas ou duas contínuas usando o método apropriado | [03 — EDA](modulos/03_eda_preparacao.md#34-comparar-duas-features) |
| 📘 | Comparar e contrastar imputação com média, mediana ou moda | [03 — EDA](modulos/03_eda_preparacao.md#35-imputação-média-mediana-ou-moda) |
| ✅ | Imputar valores faltantes com moda, média ou mediana | [03 — EDA](modulos/03_eda_preparacao.md#36-imputação-na-prática) |
| ✅ | Usar one-hot encoding para features categóricas | [05 — Feature Engineering](modulos/05_feature_engineering.md#53-one-hot-encoding) |
| 📘 | Identificar e explicar para quais tipos de modelo ou datasets o one-hot encoding é ou não apropriado | [03 — EDA](modulos/03_eda_preparacao.md#37-quando-one-hot-encoding-é-má-ideia) |
| 📘 | Identificar cenários em que a transformação em escala logarítmica é apropriada | [03 — EDA](modulos/03_eda_preparacao.md#38-transformação-logarítmica) |

---

## Seção 3 — Model Development (31% da prova)

Segunda maior seção. É onde estão as questões de cálculo (número de modelos treinados) e de julgamento (qual métrica usar).

| | Objetivo oficial | Onde estudar |
|---|---|---|
| 📘 | Usar fundamentos de ML para escolher o algoritmo adequado a um cenário | [06 — Treinamento](modulos/06_treinamento_modelos.md#61-escolher-o-algoritmo-certo) |
| 📘 | Identificar métodos para mitigar desbalanceamento de classes nos dados de treino | [06 — Treinamento](modulos/06_treinamento_modelos.md#62-desbalanceamento-de-classes) |
| 📘 | Comparar estimators e transformers | [05 — Feature Engineering](modulos/05_feature_engineering.md#51-estimators-vs-transformers) |
| ✅ | Desenvolver uma training pipeline | [05 — Feature Engineering](modulos/05_feature_engineering.md#55-pipeline-completo) |
| ✅ | Usar a operação `fmin` do Hyperopt para tunar hiperparâmetros | [07 — Tuning](modulos/07_hyperopt.md#73-hyperopt-na-prática) |
| 📘 | Executar busca aleatória, em grade ou bayesiana para tuning | [07 — Tuning](modulos/07_hyperopt.md#71-as-três-estratégias-de-busca) |
| ✅ | Paralelizar modelos single-node para tuning de hiperparâmetros | [07 — Tuning](modulos/07_hyperopt.md#76-paralelizar-o-tuning-sparktrials) |
| 📘 | Descrever benefícios e desvantagens de cross-validation vs train-validation split | [06 — Treinamento](modulos/06_treinamento_modelos.md#66-cross-validation-vs-train-validation-split) |
| ✅ | Executar cross-validation como parte do ajuste do modelo | [06 — Treinamento](modulos/06_treinamento_modelos.md#67-cross-validation-na-prática) |
| ✅ | **Identificar quantos modelos são treinados em um processo de grid-search com cross-validation** | [06 — Treinamento](modulos/06_treinamento_modelos.md#68-quantos-modelos-são-treinados--a-conta-que-cai-na-prova) |
| ✅ | Usar métricas comuns de classificação: F1, Log Loss, ROC/AUC etc. | [06 — Treinamento](modulos/06_treinamento_modelos.md#63-métricas-de-classificação) |
| ✅ | Usar métricas comuns de regressão: RMSE, MAE, R² etc. | [06 — Treinamento](modulos/06_treinamento_modelos.md#64-métricas-de-regressão) |
| 📘 | Escolher a métrica mais apropriada para o objetivo de um cenário | [06 — Treinamento](modulos/06_treinamento_modelos.md#65-qual-métrica-usar-em-cada-cenário) |
| 📘 | **Identificar a necessidade de exponenciar variáveis log-transformadas antes de calcular métricas ou interpretar predições** | [06 — Treinamento](modulos/06_treinamento_modelos.md#69-target-log-transformado-o-erro-que-quase-todo-mundo-comete) |
| 📘 | Avaliar o impacto da complexidade do modelo e do trade-off viés-variância na performance | [06 — Treinamento](modulos/06_treinamento_modelos.md#610-viés-variância-e-complexidade-do-modelo) |

---

## Seção 4 — Model Deployment (12% da prova)

A menor seção, mas com objetivos muito específicos — vale conhecer os nomes exatos.

| | Objetivo oficial | Onde estudar |
|---|---|---|
| 📘 | Identificar diferenças e vantagens entre as abordagens de serving: batch, real-time e streaming | [10 — Deployment](modulos/10_deployment.md#101-as-três-formas-de-serving) |
| ✅ | Fazer deploy de um modelo customizado em um endpoint | [10 — Deployment](modulos/10_deployment.md#106-modelo-customizado-mlflow-pyfunc) |
| ✅ | Usar pandas para fazer batch inference | [10 — Deployment](modulos/10_deployment.md#102-batch-inference-com-pandas) |
| 📘 | Identificar como a inferência em streaming é feita com Delta Live Tables | [10 — Deployment](modulos/10_deployment.md#105-streaming-inference-com-delta-live-tables) |
| 📘 | Fazer deploy e consultar um modelo para inferência em tempo real | [10 — Deployment](modulos/10_deployment.md#104-real-time-model-serving) |
| 📘 | Dividir tráfego entre endpoints para inferência em tempo real | [10 — Deployment](modulos/10_deployment.md#107-dividir-tráfego-entre-modelos-ab-testing) |

---

## Tópicos que **não** estão no exam guide atual

Muito material antigo na internet ainda cobra estes tópicos. Eles **saíram** do exam guide vigente. Saber que existem é útil no trabalho, mas não gaste tempo de estudo com eles:

| Tópico | Situação |
|---|---|
| **Stages do Model Registry** (`None → Staging → Production → Archived`) | Substituído por **aliases** no Unity Catalog. O exam guide atual só menciona aliases. |
| `transition_model_version_stage()` | API do registry legado, desabilitada por padrão em workspaces com Unity Catalog. |
| **HorovodRunner / `sparkdl`** | Removido dos runtimes recentes. Não aparece no exam guide atual. |
| **Pandas API on Spark** (`pyspark.pandas`) | Estava no exam guide antigo. Saiu da versão de março de 2025. |
| **Pandas Function APIs** (`applyInPandas`, `mapInPandas`) | Estava no exam guide antigo. Saiu. |
| **Ensembling / bagging & boosting distribuídos** como seção própria | Era a seção "Scaling ML Models" do guide antigo. Saiu. |
| **Deep learning (PyTorch/Keras) no Databricks** | Não é objetivo do guide atual desta certificação. |

> Se você encontrar um simulado de terceiros cheio de perguntas sobre `Staging`/`Production` ou `applyInPandas`, ele está baseado no exam guide antigo (anterior a março de 2025).

---

## Checklist de prontidão

Marque só quando conseguir **explicar em voz alta, sem consultar**:

```
Seção 1 — Databricks Machine Learning (38%)
□ Diferença entre Runtime padrão e Runtime ML, e o que o serverless muda
□ O que o AutoML automatiza e quais 2 notebooks ele gera
□ Por que feature tables no Unity Catalog são melhores que no workspace
□ Criar feature table: qual client, qual método, o que é obrigatório
□ Diferença entre feature table online e offline
□ log_param vs log_metric vs log_artifact vs log_model
□ Como achar o melhor run programaticamente
□ Nome de 3 níveis (catalog.schema.modelo) e por que hífen não funciona
□ Aliases: como promover challenger a champion
□ Quando promover código e quando promover modelo

Seção 2 — Data Processing (19%)
□ .summary() e o que ele retorna
□ Regra do IQR: Q1 - 1.5×IQR e Q3 + 1.5×IQR
□ Regra dos 3 desvios padrão
□ Qual gráfico para categórica, qual para contínua
□ Comparar 2 categóricas (crosstab/qui-quadrado) vs 2 contínuas (correlação/scatter)
□ Média vs mediana vs moda: quando cada uma
□ Quando one-hot encoding é ruim (árvores, alta cardinalidade)
□ Quando aplicar log (distribuição assimétrica à direita, escala multiplicativa)

Seção 3 — Model Development (31%)
□ Estimator tem fit(); transformer tem transform()
□ 4 formas de tratar desbalanceamento
□ Como contar modelos: combinações × folds (+1 se refit)
□ F1, Log Loss, ROC-AUC, PR-AUC: o que cada uma mede
□ RMSE vs MAE vs R²
□ Qual métrica escolher quando classes são desbalanceadas
□ Exponenciar o target antes de calcular a métrica em modelo log-transformado
□ Viés alto = underfitting; variância alta = overfitting
□ fmin minimiza → retornar -métrica para maximizar

Seção 4 — Model Deployment (12%)
□ Batch vs streaming vs real-time: latência, custo e caso de uso
□ Como fazer batch inference com pandas / pandas_udf
□ Como streaming inference é feito com Delta Live Tables
□ Formato do payload do endpoint (dataframe_records / dataframe_split)
□ PythonModel: predict() é obrigatório, load_context() é opcional
□ Traffic split entre served entities para A/B test
```

---

→ Voltar ao [README](README.md)
