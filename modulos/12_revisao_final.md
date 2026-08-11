# Módulo 12 — Revisão Final

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** todas
>
> **Quando fazer:** nas últimas duas semanas antes da prova.

Este módulo não ensina nada novo. Ele junta o que mais derruba candidato: data leakage, confusões de nomenclatura, armadilhas de enunciado e o checklist completo.

---

## 12.1 Data leakage — o conceito transversal

Data leakage aparece em objetivos de três seções diferentes. Vale a pena dominar.

```
Data leakage = usar, no treino, informação que não estaria disponível
               no momento real da predição.

Sintoma: métricas ótimas na validação, desempenho ruim em produção.
```

### As quatro fontes

```
1. TRANSFORMAÇÃO ANTES DO SPLIT
   ERRADO:
     scaler.fit(X_todos)              ← calcula média/desvio com dados de TESTE
     X_train, X_test = split(X_scaled)

   CORRETO:
     X_train, X_test = split(X)       ← split PRIMEIRO
     scaler.fit(X_train)              ← fit APENAS no treino
     scaler.transform(X_test)         ← transform (sem fit) no teste

   Vale para: StandardScaler, MinMaxScaler, Imputer, StringIndexer, PCA...
   → qualquer ESTIMATOR

2. FEATURE QUE EMBUTE O TARGET
   ERRADO: "ticket_medio = receita_total / n_compras"
           quando n_compras inclui a compra que você está prevendo
   Também: usar uma coluna que só é preenchida DEPOIS do evento
           (ex.: "motivo_do_cancelamento" para prever cancelamento)

3. SÉRIE TEMPORAL COM SPLIT ALEATÓRIO
   ERRADO:  train_test_split(df, test_size=0.2)     ← embaralha as datas
   CORRETO: treino = df[df.data <  "2024-01"]
            teste  = df[df.data >= "2024-01"]

4. FEATURE STORE SEM POINT-IN-TIME LOOKUP
   Usar o valor ATUAL da feature em vez do valor vigente na data do evento
   → resolver com timestamp_lookup_key (Módulo 9)
```

```python
# A defesa: encapsular tudo em um Pipeline

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)     # o scaler faz fit só em X_train
pipeline.score(X_test, y_test)     # e transform com os parâmetros do treino

# Spark ML — mesma garantia
# pipeline_model = pipeline.fit(train_df)
# pipeline_model.transform(test_df)
```

> **Por que o Pipeline é a resposta certa em tantas questões:** ele torna o leakage estruturalmente impossível. Um único `fit(train)` propaga a regra para todos os estágios.

---

## 12.2 Confusões que a prova explora

| Confusão | A verdade |
|---|---|
| `describe()` vs `summary()` | **`summary()` inclui os quartis** (25%, 50%, 75%); `describe()` não |
| Estimator vs Transformer | Estimator tem **`fit()`** e aprende. Transformer só tem `transform()` |
| `VectorAssembler` é Estimator? | **Não** — é Transformer. Só concatena colunas |
| `OneHotEncoder` é Transformer? | **Não** no Spark — é **Estimator** (precisa saber quantas categorias existem) |
| `Bucketizer` vs `QuantileDiscretizer` | Bucketizer: você define os limites, **sem `fit`**. QuantileDiscretizer: o Spark calcula, **com `fit`** |
| Nº de buckets do `Bucketizer` | N splits → **N−1 buckets** |
| `BinaryClassificationEvaluator` tem accuracy? | **Não** — só `areaUnderROC` e `areaUnderPR`. Accuracy vem do Multiclass |
| `search_runs()` retorna Spark DataFrame? | **Não** — pandas |
| `fmin()` maximiza? | **Não** — minimiza. Retorne `-métrica` para maximizar |
| `hp.choice` devolve o valor? | **Não** — devolve o **índice**. Use `space_eval()` |
| `SparkTrials` + `nested=True`? | **Não** — SparkTrials cria os runs sozinho |
| `autolog()` loga métrica de teste? | **Não** — só o que acontece dentro do `fit()` |
| `@` ou `/` no URI do modelo? | `@` = **alias** | `/` = **versão** |
| `GBTClassifier` produz `probability`? | **Sim**, desde o Spark 3. A afirmação contrária é de material desatualizado |
| `Imputer` aceita coluna de texto? | **Não** — só colunas numéricas, inclusive com `strategy="mode"` |
| Nome de modelo no UC pode ter hífen? | **Não** — só letras, números e underscore, em 3 níveis |
| Mais dados corrige underfitting? | **Não** — reduz variância (overfitting), não viés |
| `FeatureStoreClient` ou `FeatureEngineeringClient`? | **`FeatureEngineeringClient`** para Unity Catalog. O outro é legado |
| RMSE em modelo com target log? | Precisa **exponenciar** antes de calcular |

---

## 12.3 Como ler o enunciado

A maioria das questões é resolvida identificando a **palavra-chave** do enunciado:

| Palavra-chave no enunciado | Aponta para |
|---|---|
| "com o mínimo de esforço, mas correto" | Examinar os dados antes de decidir — não existe atalho automático |
| "diretamente mitiga o viés para a classe majoritária" | Class weights / cost-sensitive learning |
| "precisa explicar a decisão" | Modelo interpretável (regressão logística, árvore) |
| "melhor performance possível" | Ensemble (GBT, Random Forest) |
| "o compute precisa redimensionar dinamicamente" | Delta Live Tables |
| "usuário está esperando a resposta" | Real-time serving |
| "toda madrugada", "relatório semanal" | Batch inference |
| "governado pelo Unity Catalog" | `FeatureEngineeringClient`, nome de 3 níveis |
| "sem alterar o código de inferência" | Aliases (champion/challenger) |
| "quantos modelos serão treinados" | combinações × folds |
| "distribuição assimétrica" | Mediana (imputação), IQR (outliers), log (transformação) |
| "classes desbalanceadas" | F1 ou PR-AUC — nunca accuracy |
| "quantos buckets" | N splits → N−1 |

### Estratégia de prova

```
48 questões / 90 minutos ≈ 1 min 50 s por questão

1. Primeira passada: responda o que você sabe de imediato.
   Marque para revisão o que exigir contas ou leitura cuidadosa.
2. Segunda passada: resolva as marcadas com calma.
3. Terceira passada: revise as que ficaram em dúvida.

Nunca deixe questão em branco — não há penalidade por erro.

Nas questões de múltipla seleção, o enunciado diz quantas marcar
("Selecione DUAS"). Não há crédito parcial: marque o número exato.
```

---

## 12.4 Glossário

| Termo | Significado |
|---|---|
| **Alias** | Ponteiro nomeado para uma versão de modelo no UC (`champion`, `challenger`) |
| **Autolog** | Logging automático do MLflow durante o `fit()` |
| **Bagging** | Treinar modelos em paralelo sobre amostras e agregar. Reduz **variância** (Random Forest) |
| **Boosting** | Treinar modelos em sequência, cada um corrigindo o anterior. Reduz **viés** (GBT) |
| **Champion / Challenger** | Modelo em produção / candidato a substituí-lo |
| **Concept drift** | A relação entre features e target mudou |
| **Data drift** | A distribuição das features de entrada mudou |
| **Deploy code vs deploy model** | Promover o código de treino vs promover o binário treinado |
| **Estimator** | Componente com `fit()` — aprende dos dados |
| **Feature table** | Tabela Delta com primary key, gerenciada pelo Feature Store |
| **Glass-box** | O AutoML entrega o código-fonte do modelo, não um binário opaco |
| **IQR** | Q3 − Q1. Outlier: fora de `Q1 − 1.5×IQR` a `Q3 + 1.5×IQR` |
| **Lazy evaluation** | Transformações só executam quando uma action é chamada |
| **Log Loss** | Perda logarítmica. Menor é melhor. Penaliza confiança errada |
| **MLOps** | Versionar e operar código + dados + modelos |
| **Point-in-time lookup** | Buscar o valor da feature vigente na data do evento |
| **`pyfunc`** | Formato universal de modelo do MLflow |
| **Run** | Uma execução de treino registrada no MLflow |
| **Serverless** | Compute gerenciado pelo Databricks, sem configuração |
| **Training-serving skew** | Divergência entre features do treino e de produção |
| **Transformer** | Componente com `transform()` — não aprende nada |
| **TPE** | Tree of Parzen Estimators — a otimização bayesiana do Hyperopt |
| **Unity Catalog** | Governança em 3 níveis: `catalog.schema.objeto` |

---

## 12.5 Checklist final por seção

### Seção 1 — Databricks Machine Learning (38%)

```
MLflow
□ log_param / log_params, log_metric / log_metrics, log_artifact, log_model
□ log_dict / log_text / log_figure / log_table criam o arquivo; log_artifact exige que exista
□ autolog(): o que registra e o que NÃO registra (métricas de teste)
□ search_runs() retorna DataFrame pandas
□ MlflowClient().search_runs(order_by=[...], max_results=1) → melhor run
□ MLflow 3 usa name= em log_model (artifact_path= depreciado)
□ start_run(nested=True) — para Hyperopt com Trials()
□ log_metric com step → curvas na UI
□ O que a MLflow UI mostra: params, métricas, artefatos, assinatura, tags, comparação

Registry no Unity Catalog
□ set_registry_uri("databricks-uc")
□ Nome de 3 níveis, sem hífen
□ register_model() / create_model_version() via Client API
□ set_registered_model_alias() = promover
□ get_model_version_by_alias()
□ Tags: set/delete em versão, em modelo registrado e em run
□ models:/nome@alias (alias) vs models:/nome/3 (versão)
□ Benefícios do UC sobre workspace registry: escopo de conta, governança, linhagem, aliases

Feature Store
□ FeatureEngineeringClient (não FeatureStoreClient)
□ create_table(name, primary_keys, df ou schema)
□ write_table(mode="merge" | "overwrite")
□ FeatureLookup(table_name, feature_names, lookup_key)
□ create_training_set(df, feature_lookups, label, exclude_columns)
□ fe.log_model() vs mlflow.sklearn.log_model()
□ fe.score_batch() recebe só as chaves
□ timestamp_lookup_key → point-in-time
□ Online vs offline: latência, volume, uso, histórico
□ Benefícios de feature tables no UC vs no workspace

AutoML
□ classify / regress / forecast
□ primary_metric define o melhor modelo
□ forecast exige time_col, frequency, horizon
□ Gera 2 notebooks: EDA + melhor modelo editável
□ Glass-box: entrega o código, não um binário
□ Automatiza seleção de features e de modelos
□ Classificação/regressão exigem compute clássico com Runtime ML

MLOps
□ Versionar código + dados + modelo
□ Deploy code é o padrão; deploy model quando treino é caro ou há exigência de auditoria
□ Champion / challenger
□ Data drift vs concept drift

Plataforma
□ Vantagens do Runtime ML: libs prontas, versões validadas, start rápido, MLflow integrado
□ Job Cluster para produção; All-Purpose para desenvolvimento
□ Delta Lake: ACID, time travel (versionAsOf / timestampAsOf), DESCRIBE HISTORY
```

### Seção 2 — Data Processing (19%)

```
□ describe() vs summary() (quartis!)
□ dbutils.data.summarize()
□ approxQuantile() para calcular quartis
□ Outliers por IQR: Q1 − 1.5×IQR, Q3 + 1.5×IQR
□ Outliers por desvio padrão: média ± 3σ
□ Por que IQR é mais robusto (média e desvio são distorcidos pelos outliers)
□ Gráfico por tipo: histograma/boxplot (contínua), barras (categórica), scatter (2 contínuas)
□ 2 contínuas → correlação | 2 categóricas → crosstab + qui-quadrado
□ Pearson mede relação LINEAR; zero não significa "sem relação"
□ Imputação: simétrica → média | assimétrica → mediana | categórica → moda
□ Imputer do Spark só aceita colunas numéricas
□ Quando OHE é ruim: árvores, alta cardinalidade, variáveis ordinais
□ dropLast=True evita multicolinearidade
□ Log: assimetria à direita, crescimento multiplicativo, ordens de grandeza
□ log1p ↔ expm1; árvores são indiferentes ao log
```

### Seção 3 — Model Development (31%)

```
□ Estimator vs Transformer — saber classificar cada componente
□ Ordem: StringIndexer → OneHotEncoder → VectorAssembler
□ handleInvalid: error | skip | keep
□ Pipeline.fit() → PipelineModel
□ Escolha de algoritmo: interpretabilidade vs performance
□ Desbalanceamento: class weights, over/undersampling, threshold, troca de métrica
□ weightCol no Spark (não existe class_weight="balanced")
□ Accuracy, precision, recall, F1 — quando cada uma
□ Log Loss: menor é melhor, exige probabilidades
□ ROC-AUC vs PR-AUC em classes desbalanceadas
□ RMSE vs MAE vs R²
□ BinaryClassificationEvaluator só tem areaUnderROC e areaUnderPR
□ CrossValidator vs TrainValidationSplit: prós e contras
□ NÚMERO DE MODELOS = combinações × folds
□ Exponenciar target log-transformado antes de calcular métricas
□ Viés alto = underfitting | variância alta = overfitting
□ Mais dados reduz variância, não viés
□ Bagging reduz variância; boosting reduz viés
□ fmin minimiza → retornar -métrica
□ hp.choice → índice → space_eval
□ hp.quniform → float → int()
□ hp.loguniform recebe limites JÁ em log
□ tpe.suggest (bayesiana) vs rand.suggest (aleatória) vs grid
□ Trials() vs SparkTrials(parallelism=N)
□ SparkTrials para modelos SINGLE-NODE; Trials() para modelos distribuídos
```

### Seção 4 — Model Deployment (12%)

```
□ Batch vs streaming vs real-time: latência, custo, caso de uso
□ Batch com pandas / toPandas() só em amostras
□ pandas_udf: vetorizada, opera em chunks, roda nos workers
□ mlflow.pyfunc.spark_udf() → modelo como UDF Spark
□ Streaming com Delta Live Tables: modelo como UDF, autoscaling automático
□ Anti-padrão: chamar endpoint REST por evento em stream de alto volume
□ checkpointLocation é obrigatório em writeStream
□ Payload: dataframe_records ou dataframe_split
□ PythonModel: predict() obrigatório, load_context() opcional
□ Modelo customizado vai para endpoint como qualquer outro
□ Traffic split: served entities no MESMO endpoint, somando 100%
□ Canary, A/B, shadow, blue-green
```

---

## 12.6 Simulado final

Quando você marcar todos os itens do checklist:

1. Rode o **Modo Prova** do simulado: 48 questões, 90 minutos, sem consultar nada
2. Meta: **≥ 80%** em três simulados completos, em dias diferentes
3. Para cada erro, volte ao módulo correspondente — não decore a resposta, entenda o porquê
4. Releia o [mapa dos objetivos oficiais](../GUIA_DA_PROVA.md) e confirme que consegue explicar cada linha em voz alta

```bash
streamlit run simulados/quiz.py
```

---

## 12.7 Duas semanas antes da prova

- [ ] Baixe o **exam guide oficial** e compare com o [mapa deste guia](../GUIA_DA_PROVA.md) — o Databricks atualiza o conteúdo sem aviso
- [ ] Faça o **system check** da plataforma de proctoring (webcam, microfone, internet)
- [ ] Confirme o **idioma** da prova na inscrição — existe versão em Português (BR)
- [ ] Prepare o ambiente físico: mesa limpa, documento com foto, sem outras pessoas na sala
- [ ] Lembre: **não é permitido material de consulta**

> ⚠️ **Lembrete final:** este guia é um material **experimental, não oficial e sem garantia de aprovação** — ver o [aviso completo no README](../README.md#️-aviso-importante--leia-antes-de-começar). Use-o como complemento ao exam guide oficial e à documentação da Databricks, nunca como substituto.

---

→ Voltar ao [README](../README.md) · [Mapa dos objetivos oficiais](../GUIA_DA_PROVA.md)
