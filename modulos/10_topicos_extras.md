# Módulo 10 — Tópicos Extras da Prova

> **Notebook sugerido:** `10_topicos_extras`
>
> Tópicos adicionais que aparecem nas questões: data leakage, deep learning, métricas e checklist final.

---

## 10.1 Data Leakage — erro crítico que cai na prova

```
Data Leakage = usar informação do futuro durante o treinamento.
Resultado: modelo parece ótimo no treino/validação, péssimo em produção.
```

### Fontes comuns de leakage

```
1. Normalização antes do split
   ERRADO:
     scaler.fit(X_todos)          ← usa estatísticas do conjunto de TESTE!
     X_train, X_test = split(X_todos_scaled)

   CORRETO:
     X_train, X_test = split(X)   ← split PRIMEIRO
     scaler.fit(X_train)          ← fit APENAS no treino
     scaler.transform(X_test)     ← transform (sem fit) no teste

2. Features calculadas com dados do target
   ERRADO: "receita_media = total_receita / n_compras"
           onde n_compras inclui a compra que você está prevendo

3. Séries temporais — split aleatório
   ERRADO:  train_test_split(df, test_size=0.2)  ← mistura datas
   CORRETO: df[df.data < "2024-01"].train / df[df.data >= "2024-01"].test

4. Feature Store sem point-in-time lookup
   Usar valor atual da feature ao invés do valor que existia na data do evento
```

```python
# ERRADO: data leakage na normalização
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Normaliza ANTES de separar treino/teste → vazamento das stats do teste para o treino!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)           # usa X_test para calcular média/desvio!
X_train, X_test = train_test_split(X_scaled)  # ← leakage

# CORRETO: usar Pipeline do sklearn (encapsula e garante o fluxo)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)   # scaler.fit() usa só X_train
pipeline.score(X_test, y_test)   # scaler.transform() usa parâmetros do treino

# MELHOR AINDA: Spark ML Pipeline (já garante isso)
# Pipeline.fit(train_df) → todos os estágios aprendem só no treino
# pipeline_model.transform(test_df) → aplica parâmetros do treino no teste
```

---

## 10.2 Métricas — referência completa

### Classificação Binária

```python
from sklearn.metrics import (
    accuracy_score,           # (TP+TN) / total — ⚠️ enganoso em datasets desbalanceados
    precision_score,          # TP / (TP+FP) — "dos previstos como positivo, quantos realmente são?"
    recall_score,             # TP / (TP+FN) — "dos positivos reais, quantos eu capturei?"
    f1_score,                 # média harmônica de precision e recall
    roc_auc_score,            # área sob a curva ROC [0.5 = random, 1.0 = perfeito]
    average_precision_score,  # AUC-PR — melhor que ROC para datasets desbalanceados
    log_loss,                 # perda logarítmica — penaliza predições confiantes e erradas
)

# Spark ML equivalentes:
# BinaryClassificationEvaluator → "areaUnderROC", "areaUnderPR"
# MulticlassClassificationEvaluator → "accuracy", "f1", "weightedPrecision", "weightedRecall"
```

### Regressão

```python
from sklearn.metrics import (
    mean_squared_error,    # MSE — penaliza erros grandes (sensível a outliers)
    mean_absolute_error,   # MAE — robusto a outliers
    r2_score,              # R² ∈ (-∞, 1] — 1.0 = perfeito, 0 = mesma performance que média
)
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE na mesma unidade do target

# Spark ML — RegressionEvaluator → "rmse", "mse", "mae", "r2", "var"
```

### Quando usar qual?

```
Dados balanceados:       accuracy ou f1
Dados desbalanceados:    f1, AUC-ROC, AUC-PR (evitar accuracy!)
Custo de falso positivo alto:   priorizar precision
Custo de falso negativo alto:   priorizar recall
Regressão com outliers:  MAE (mais robusto que RMSE)
Regressão comparativa:   R² (normalizado, mais interpretável)
```

---

## 10.3 Deep Learning no Databricks

### PyTorch com MLflow autolog

```python
import torch
import torch.nn as nn
import mlflow.pytorch

class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

import mlflow

with mlflow.start_run(run_name="pytorch_churn"):
    mlflow.pytorch.autolog()   # loga loss, modelo, etc. automaticamente

    model_nn = ChurnNet(input_dim=3)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X_train.values)
    y_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)

    for epoch in range(50):
        optimizer.zero_grad()
        output = model_nn(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("train_loss", loss.item(), step=epoch)
```

### Keras/TensorFlow com MLflow autolog

```python
import tensorflow as tf
import mlflow.keras

with mlflow.start_run(run_name="keras_churn"):
    mlflow.keras.autolog()   # loga loss, acurácia por época, modelo

    model_keras = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model_keras.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model_keras.fit(
        X_train.values, y_train.values,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
```

---

## 10.4 Treinamento Distribuído — HorovodRunner

```python
# HorovodRunner: treinar modelos de deep learning em múltiplas GPUs/máquinas
from sparkdl import HorovodRunner

def train_hvd():
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf

    hvd.init()

    # Ajustar learning rate para treinamento distribuído
    lr = 0.001 * hvd.size()   # escala LR com o número de workers
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    optimizer = hvd.DistributedOptimizer(optimizer)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),  # sincroniza pesos
        hvd.callbacks.MetricAverageCallback(),               # média das métricas
    ]

    model.fit(X_train, y_train, epochs=50, callbacks=callbacks)

    # Salvar apenas no worker principal (rank 0)
    if hvd.rank() == 0:
        mlflow.keras.log_model(model, "model")

runner = HorovodRunner(np=2)   # np = número de workers/GPUs
runner.run(train_hvd)
```

---

## 10.5 KMeans Clustering com Spark ML

```python
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# KMeans
kmeans = KMeans(
    featuresCol="features",
    k=3,           # número de clusters
    maxIter=20,
    seed=42
)

df = spark.read.format("delta").load("dbfs:/FileStore/churn_project/data/clientes")
# (precisaria ter a coluna "features" — usando a do pipeline do M4)

km_model  = kmeans.fit(train_feat)
clustered = km_model.transform(test_feat)
# clustered tem coluna "prediction" com o ID do cluster (0, 1, 2)

# Avaliar com Silhouette
evaluator = ClusteringEvaluator(
    featuresCol="features",
    metricName="silhouette"  # [-1, 1] — maior é melhor
)
silhouette = evaluator.evaluate(clustered)
print(f"Silhouette score: {silhouette:.4f}")

# Centros dos clusters
print("Centros dos clusters:")
for i, center in enumerate(km_model.clusterCenters()):
    print(f"  Cluster {i}: {center}")
```

---

## 10.6 Checklist Final — O que revisar antes da prova

```
MLflow (25% da prova):
□ log_param / log_params, log_metric / log_metrics, log_model, log_artifact
□ autolog() — o que registra automaticamente, para quais frameworks
□ start_run(nested=True) — para que serve (Hyperopt!)
□ Stages do Registry: None → Staging → Production → Archived
□ Carregar por stage: "models:/nome/Production"
□ Carregar por versão: "models:/nome/2"
□ MlflowClient para operações programáticas
□ search_runs() retorna DataFrame **pandas** (não Spark!)
□ log_metric com step — para curvas de aprendizado

Spark ML (20% da prova):
□ VectorAssembler → coluna "features" do tipo Vector (obrigatória)
□ Ordem: StringIndexer → OneHotEncoder → VectorAssembler
□ handleInvalid: "error" | "skip" | "keep"
□ Pipeline.fit(train) → PipelineModel
□ BinaryClassificationEvaluator → "areaUnderROC" ou "areaUnderPR"
□ MulticlassClassificationEvaluator → "accuracy", "f1", "weightedPrecision", "weightedRecall"
□ RegressionEvaluator → "rmse", "mse", "mae", "r2"
□ CrossValidator vs TrainValidationSplit (k-fold vs split único)
□ GBTClassifier NÃO produz coluna "probability"

Hyperopt:
□ fmin() minimiza → retornar -métrica para maximizar
□ hp.choice → índice (usar space_eval para converter)
□ hp.quniform → float (converter com int())
□ hp.loguniform → para learning rate e regularização
□ Trials() sequencial vs SparkTrials(parallelism=N) paralelo
□ SparkTrials NÃO suporta nested=True

AutoML:
□ 3 tipos: classify, regress, forecast
□ primary_metric controla qual modelo é "melhor"
□ forecast precisa de: time_col, frequency, horizon
□ Gera 2 notebooks: EDA + melhor modelo (editável)
□ NÃO disponível na Community Edition

Feature Store:
□ primary_keys → chave de lookup (join)
□ FeatureLookup → especifica features, tabela e lookup_key
□ fs.log_model() (não mlflow.sklearn.log_model) para vincular ao FS
□ fs.score_batch() busca features automaticamente
□ mode="merge" → upsert; mode="overwrite" → substitui tudo
□ timestamp_lookup_key → point-in-time lookup

Deployment:
□ 3 formas: batch, streaming, real-time serving
□ pyfunc = formato universal MLflow
□ pandas_udf = UDF vetorizada para Spark (eficiente para grandes volumes)
□ Endpoint aceita: "dataframe_records" ou "dataframe_split"
□ PythonModel implementa: load_context() e predict()

Delta Lake:
□ ACID, time travel, schema enforcement
□ option("versionAsOf", N) → versão específica
□ DESCRIBE HISTORY → histórico de versões
□ mode: "overwrite", "append", "merge"

Data Leakage:
□ Sempre split ANTES de fit qualquer transformador
□ Pipeline garante que fit() usa apenas dados de treino
□ Séries temporais: split temporal (não aleatório)
□ Feature Store com point-in-time para evitar leakage temporal
```

---

→ Voltar ao: [README.md](../README.md) — Índice completo e plano de estudos
