# Módulo 9 — Deployment e Serving

> **Notebook sugerido:** `09_deployment`
>
> Depois de treinar o modelo, existem 3 formas de servir predições no Databricks.

> **Pré-requisito:** este módulo carrega um modelo já registrado como `workspace.default.churn_predictor` com o alias `champion` (seção 3.6 do Módulo 3) e a tabela `workspace.default.churn_clientes` (Módulo 2). Rode os Módulos 2 e 3 antes de continuar — sem isso, `mlflow.pyfunc.load_model(...)` abaixo vai falhar por o modelo/alias não existir.

---

## 9.1 As três formas de serving

```
1. Batch Inference   → aplica o modelo em grandes volumes de dados de uma vez
                       Ex: pontuar 1 milhão de clientes toda noite
                       ✅ Disponível na Free Edition

2. Streaming         → aplica em dados chegando em tempo real (Spark Structured Streaming)
                       Ex: pontuar transações à medida que chegam
                       ✅ Disponível na Free Edition

3. Real-time Serving → endpoint REST para predições em milissegundos
                       Ex: API chamada pelo app móvel para prever churn em tempo real
                       ✅ Disponível na Free Edition, com limites (poucos endpoints ativos, sem GPU)
```

---

## 9.2 Batch Inference — opção 1: pyfunc direto

```python
import mlflow
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import functions as F

# Carregar modelo do Registry (formato genérico pyfunc)
# Nome UC: catalog.schema.modelo (sem hífen) — "champion" é o alias definido no Módulo 3
MODEL_NAME = "workspace.default.churn_predictor"
model_uri  = f"models:/{MODEL_NAME}@champion"

# pyfunc = formato universal — funciona com sklearn, XGBoost, Keras, PyTorch...
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# Para datasets pequenos: converter para pandas e predizer
df = spark.table("workspace.default.churn_clientes")
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

df_small_pd = df.select(features_cols).limit(100).toPandas()
preds = pyfunc_model.predict(df_small_pd)
print(f"Primeiras predições: {preds[:10]}")
```

---

## 9.3 Batch Inference — opção 2: Pandas UDF (eficiente para grandes volumes)

```python
from pyspark.sql.functions import pandas_udf, struct
from pyspark.sql.types import DoubleType
import pandas as pd

# Carregar modelo no driver (uma vez)
loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

# Pandas UDF: aplica o modelo em chunks pandas distribuídos nos workers
# Muito mais eficiente que converter tudo para pandas de uma vez
@pandas_udf(DoubleType())
def predict_churn_udf(*feature_series: pd.Series) -> pd.Series:
    df_features = pd.concat(feature_series, axis=1)
    df_features.columns = features_cols
    # Retorna probabilidade de churn (classe 1)
    return pd.Series(loaded_model.predict_proba(df_features)[:, 1])

# Aplicar no DataFrame Spark inteiro (distribuído!)
df = spark.table("workspace.default.churn_clientes")

df_com_score = df.withColumn(
    "prob_churn",
    predict_churn_udf(*[F.col(c) for c in features_cols])
)

df_com_score = df_com_score.withColumn(
    "predicao_churn",
    F.when(F.col("prob_churn") >= 0.5, 1).otherwise(0)
)

display(
    df_com_score
    .select("customer_id", "prob_churn", "predicao_churn", "churn")
    .orderBy(F.col("prob_churn").desc())
    .limit(15)
)
```

---

## 9.4 Salvar predições em Delta

```python
# Pipeline completo de batch inference → salvar resultado

TABELA_PREDICOES = "workspace.default.churn_predictions_batch"

(df_com_score
    .select(
        "customer_id",
        F.round("prob_churn", 4).alias("probabilidade_churn"),
        "predicao_churn",
        F.current_timestamp().alias("scored_at"),
        F.lit(MODEL_NAME).alias("modelo_usado"),
        F.lit("champion").alias("alias_modelo")
    )
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_PREDICOES)
)

print(f"Predições salvas em: {TABELA_PREDICOES}")

# Ler e verificar
df_preds = spark.table(TABELA_PREDICOES)
display(df_preds.limit(10))
print(f"Total: {df_preds.count()} clientes pontuados")
print(f"Churn previsto: {df_preds.filter(F.col('predicao_churn')==1).count()}")
```

---

## 9.5 Real-time Model Serving

> ✅ **Disponível no Free Edition** (diferente da antiga Community Edition) — com limites: número de endpoints ativos limitado, sem GPU serving, sem provisioned throughput, sem modelos customizados em GPU/batch inference. Dá pra criar e testar o endpoint abaixo direto na sua conta gratuita.

```
Na UI do Databricks:
Machine Learning → Serving → Create serving endpoint

Configurações:
├── Name: churn-predictor-endpoint
├── Served entities:
│   ├── Model: workspace.default.churn_predictor (do Unity Catalog)
│   ├── Model version: 1 (ou selecione pelo alias "champion")
│   └── Compute size: Small / Medium / Large
└── Enable scale to zero: sim (economiza quando não há tráfego)
```

```python
# Chamar o endpoint via REST API
import requests
import json

WORKSPACE_URL = "https://<seu-workspace>.cloud.databricks.com"  # copie da barra de endereço do seu navegador

# Token: buscar via dbutils.secrets ou variável de ambiente (nunca hardcoded!)
# Para testar rapidamente: Settings → Developer → Access tokens → Generate new token
TOKEN = dbutils.secrets.get(scope="meu-scope", key="api-token")

endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/churn-predictor-endpoint/invocations"

# Formato 1: dataframe_records (mais comum)
payload_records = {
    "dataframe_records": [
        {"tenure_months": 3,  "monthly_charges": 85.0, "total_charges": 255.0},
        {"tenure_months": 48, "monthly_charges": 50.0, "total_charges": 2400.0},
    ]
}

# Formato 2: dataframe_split
payload_split = {
    "dataframe_split": {
        "columns": ["tenure_months", "monthly_charges", "total_charges"],
        "data": [[3, 85.0, 255.0], [48, 50.0, 2400.0]]
    }
}

response = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=payload_records
)
print(response.json())
# {"predictions": [1, 0]}   → 1=churn, 0=fica
```

---

## 9.6 Streaming Inference

```python
from pyspark.sql import functions as F
import mlflow

# Ler stream de novos clientes chegando em tempo real
df_stream = (spark
    .readStream
    .option("readChangeFeed", "true")   # lê apenas mudanças novas
    .table("workspace.default.churn_clientes")
)

# Aplicar modelo no stream (mesma lógica do batch)
loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

@pandas_udf(DoubleType())
def predict_stream_udf(*cols: pd.Series) -> pd.Series:
    df_f = pd.concat(cols, axis=1)
    df_f.columns = features_cols
    return pd.Series(loaded_model.predict_proba(df_f)[:, 1])

df_scored_stream = df_stream.withColumn(
    "prob_churn",
    predict_stream_udf(*[F.col(c) for c in features_cols])
)

# Escrever resultado do stream numa tabela gerenciada (checkpoint fica num Volume)
query = (df_scored_stream
    .select("customer_id", "prob_churn")
    .writeStream
    .outputMode("append")
    .option("checkpointLocation", "/Volumes/workspace/default/churn_project/checkpoints/stream")
    .toTable("workspace.default.churn_predictions_stream")
)
```

---

## 9.7 MLflow pyfunc customizado

```python
import mlflow.pyfunc
import pandas as pd
import numpy as np
import pickle

class ChurnModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper que adiciona regras de negócio ao modelo sklearn.
    """

    def load_context(self, context):
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)
        self.threshold = 0.4  # threshold de negócio (não 0.5 padrão)

    def predict(self, context, model_input):
        proba = self.model.predict_proba(model_input)[:, 1]

        return pd.DataFrame({
            "prob_churn":        proba,
            "predicao":          (proba >= self.threshold).astype(int),
            "acao_recomendada":  np.where(
                proba >= 0.7, "Contato urgente",
                np.where(proba >= 0.4, "Oferta de retencao", "Monitorar")
            )
        })


# Salvar e registrar o wrapper
import tempfile, os

model_final = loaded_model   # usando o modelo carregado anteriormente

with tempfile.TemporaryDirectory() as tmp:
    pkl_path = os.path.join(tmp, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_final, f)

    with mlflow.start_run(run_name="churn_pyfunc_wrapper"):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChurnModelWrapper(),
            artifacts={"model_path": pkl_path},
            registered_model_name="workspace.default.churn_wrapper"  # nome UC: sem hífen
        )

# Usar o wrapper (por versão — não precisa de alias aqui)
wrapper = mlflow.pyfunc.load_model("models:/workspace.default.churn_wrapper/1")

resultado = wrapper.predict(pd.DataFrame({
    "tenure_months":   [2, 60, 15],
    "monthly_charges": [90.0, 45.0, 70.0],
    "total_charges":   [180.0, 2700.0, 1050.0]
}))
display(spark.createDataFrame(resultado))
```

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| **Batch Inference** | `mlflow.pyfunc.load_model()` → prediz em pandas; `pandas_udf` para escala Spark |
| **Streaming** | Spark Structured Streaming + `pandas_udf` |
| **Real-time Serving** | Endpoint REST — disponível no Free Edition, com limites (endpoints, GPU, throughput) |
| `mlflow.pyfunc.load_model()` | Formato universal — funciona com qualquer framework |
| `pandas_udf` | UDF vetorizada: opera em chunks pandas, muito mais eficiente |
| Formato do endpoint | `{"dataframe_records": [...]}` ou `{"dataframe_split": {...}}` |
| `PythonModel` | Interface para criar wrappers pyfunc customizados |
| Método obrigatório | `predict(self, context, model_input)` — deve ser implementado |
| `load_context()` | Carrega artefatos (modelos, configs) na inicialização do wrapper |

---

→ Próximo: [10_topicos_extras.md](10_topicos_extras.md)
