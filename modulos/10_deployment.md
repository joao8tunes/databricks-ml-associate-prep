# Módulo 10 — Deployment e Serving

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 4 — Model Deployment (**12% da prova**)
>
> **Notebook sugerido:** `10_deployment`
>
> **Pré-requisito:** Módulos 2 e 4 — este módulo carrega o modelo `workspace.default.churn_predictor` com o alias `champion`, registrado na seção 4.7. Sem ele, `load_model(...)` falha.

**Objetivos oficiais cobertos neste módulo:**
- Identificar as diferenças e vantagens entre batch, real-time e streaming
- Usar pandas para fazer batch inference
- Identificar como a inferência em streaming é feita com Delta Live Tables
- Fazer deploy e consultar um modelo para inferência em tempo real
- Fazer deploy de um modelo customizado em um endpoint
- Dividir tráfego entre endpoints para inferência em tempo real

---

## 10.1 As três formas de serving

```
1. BATCH INFERENCE
   Aplica o modelo em um grande volume de dados, de uma vez, periodicamente.
   Ex.: pontuar 1 milhão de clientes toda madrugada e gravar em tabela.
   ├── Latência: minutos a horas (a predição já está pronta quando é consultada)
   ├── Throughput: altíssimo
   ├── Custo: o mais baixo por predição
   └── ✅ Roda na Free Edition

2. STREAMING INFERENCE
   Aplica o modelo em dados que chegam continuamente.
   Ex.: pontuar transações à medida que entram na fila.
   ├── Latência: segundos a minutos
   ├── Throughput: alto e contínuo
   ├── Custo: médio (o cluster fica ligado)
   └── ✅ Roda na Free Edition (com limites de pipeline)

3. REAL-TIME SERVING
   Endpoint REST que responde a uma requisição por vez.
   Ex.: o app mobile pergunta "este cliente vai cancelar?" e espera a resposta.
   ├── Latência: milissegundos
   ├── Throughput: limitado pelo endpoint
   ├── Custo: o mais alto por predição
   └── ✅ Roda na Free Edition, com limites (poucos endpoints, sem GPU)
```

### Como escolher — o que a prova pergunta

| O enunciado diz… | Resposta |
|---|---|
| "todo dia à noite", "relatório semanal", "pontuar toda a base" | **Batch** |
| "usuário está esperando a resposta", "aplicativo", "API", "milissegundos" | **Real-time** |
| "eventos chegando continuamente", "à medida que os dados chegam", "janela de tempo" | **Streaming** |
| "menor custo por predição, latência não importa" | **Batch** |
| "volume flutua ao longo do dia e o compute precisa redimensionar sozinho" | **Streaming com Delta Live Tables** |

> **A pergunta de decisão:** *quando a predição é necessária?* Se pode ser calculada com antecedência e consultada depois → batch. Se depende de uma entrada que só existe no instante da requisição → real-time. Se os dados chegam num fluxo contínuo → streaming.

---

## 10.2 Batch inference com pandas

Objetivo oficial: "usar pandas para fazer batch inference".

```python
import mlflow
import pandas as pd
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = "workspace.default.churn_predictor"     # nome UC, sem hífen
MODEL_URI  = f"models:/{MODEL_NAME}@champion"        # @ = alias (ver Módulo 4)

# pyfunc = formato universal: funciona com sklearn, XGBoost, Keras, PyTorch...
pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)

df = spark.table("workspace.default.churn_clientes")
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

# Dataset pequeno: traga para pandas e prediga direto
df_pd = df.select(features_cols).limit(100).toPandas()
preds = pyfunc_model.predict(df_pd)

print("Primeiras predições:", preds[:10])
```

> ⚠️ **`toPandas()` traz tudo para a memória do driver.** Serve para amostras e datasets pequenos. Para milhões de linhas, use a `pandas_udf` da próxima seção — senão o driver estoura a memória.

---

## 10.3 Batch em escala: `pandas_udf`

```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd

# Carrega o modelo uma vez; ele é serializado para os workers
loaded_model = mlflow.sklearn.load_model(MODEL_URI)
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

# pandas_udf: aplica o modelo em CHUNKS pandas distribuídos nos workers.
# Muito mais eficiente que uma UDF Python comum, que roda linha a linha.
@pandas_udf(DoubleType())
def predict_churn_udf(*feature_series: pd.Series) -> pd.Series:
    df_features = pd.concat(feature_series, axis=1)
    df_features.columns = features_cols
    return pd.Series(loaded_model.predict_proba(df_features)[:, 1])

df = spark.table("workspace.default.churn_clientes")

df_com_score = (df
    .withColumn("prob_churn", predict_churn_udf(*[F.col(c) for c in features_cols]))
    .withColumn("predicao_churn", F.when(F.col("prob_churn") >= 0.5, 1).otherwise(0))
)

display(
    df_com_score
    .select("customer_id", "prob_churn", "predicao_churn", "churn")
    .orderBy(F.col("prob_churn").desc())
    .limit(15)
)
```

| | UDF Python comum | `pandas_udf` (vetorizada) |
|---|---|---|
| Unidade de processamento | Uma linha por vez | Um batch (Series pandas) |
| Serialização | Python ↔ JVM por linha | Apache Arrow, em blocos |
| Performance | Lenta | **Muito mais rápida** |
| Uso com sklearn/numpy | Ineficiente | Natural (vetorizado) |

### Gravar as predições

```python
TABELA_PREDICOES = "workspace.default.churn_predictions_batch"

(df_com_score
    .select(
        "customer_id",
        F.round("prob_churn", 4).alias("probabilidade_churn"),
        "predicao_churn",
        F.current_timestamp().alias("scored_at"),
        F.lit(MODEL_NAME).alias("modelo_usado"),
        F.lit("champion").alias("alias_modelo"),
    )
    .write.format("delta").mode("overwrite")
    .saveAsTable(TABELA_PREDICOES))

df_preds = spark.table(TABELA_PREDICOES)
print(f"Total pontuado: {df_preds.count()}")
print(f"Churn previsto: {df_preds.filter(F.col('predicao_churn') == 1).count()}")
display(df_preds.limit(10))
```

> **Boa prática que a prova valoriza:** gravar junto de cada predição o **modelo e a versão/alias usados** e o **timestamp**. Sem isso, é impossível auditar depois por que uma decisão foi tomada.

---

## 10.4 Real-time Model Serving

> ✅ Disponível na Free Edition, com limites: número reduzido de endpoints ativos, sem GPU serving, sem provisioned throughput.

### Criar o endpoint pela interface

```
Menu esquerdo → "Serving" → "Create serving endpoint"

├── Name: churn-predictor-endpoint
├── Entity: workspace.default.churn_predictor (do Unity Catalog)
├── Version: selecione a versão OU o alias "champion"
├── Compute size: Small / Medium / Large
└── Scale to zero: ativado (economiza quando não há tráfego)
```

### Consultar o endpoint

```python
import requests

# A URL do workspace: copie da barra de endereço do navegador
WORKSPACE_URL = "https://<seu-workspace>.cloud.databricks.com"

# NUNCA coloque o token no código. Use dbutils.secrets:
#   Settings → Developer → Access tokens → Generate new token
#   depois grave em um secret scope
TOKEN = dbutils.secrets.get(scope="meu-scope", key="api-token")

endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/churn-predictor-endpoint/invocations"

# Formato 1: dataframe_records — lista de dicionários, mais legível
payload_records = {
    "dataframe_records": [
        {"tenure_months": 3,  "monthly_charges": 85.0, "total_charges": 255.0},
        {"tenure_months": 48, "monthly_charges": 50.0, "total_charges": 2400.0},
    ]
}

# Formato 2: dataframe_split — colunas e dados separados, mais compacto
payload_split = {
    "dataframe_split": {
        "columns": ["tenure_months", "monthly_charges", "total_charges"],
        "data": [[3, 85.0, 255.0], [48, 50.0, 2400.0]],
    }
}

resposta = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=payload_records,
)
print(resposta.json())     # {"predictions": [1, 0]}
```

| Formato do payload | Estrutura |
|---|---|
| `dataframe_records` | `[{"col": valor, ...}, ...]` — uma lista de dicionários |
| `dataframe_split` | `{"columns": [...], "data": [[...], ...]}` |
| `instances` / `inputs` | Formatos tensoriais, para modelos de deep learning |

> **Scale to zero:** o endpoint desliga sozinho sem tráfego e reduz o custo a zero. O preço é o **cold start**: a primeira requisição depois da hibernação demora bem mais. Para latência garantida, mantenha desativado.

---

## 10.5 Streaming inference com Delta Live Tables

Objetivo oficial: "identificar como a inferência em streaming é feita com Delta Live Tables". Este é conceitual — e é a **Questão 5 do exam guide oficial**.

> **Nomenclatura:** o Databricks renomeou Delta Live Tables (DLT) para **Lakeflow Declarative Pipelines**. O exam guide vigente ainda usa "Delta Live Tables". Os dois nomes se referem à mesma coisa.

### O padrão

```
1. O modelo é carregado do MLflow e registrado como uma Spark UDF
       modelo_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

2. Uma pipeline DLT define tabelas em streaming
       @dlt.table
       def eventos_pontuados():
           return spark.readStream.table("...").withColumn("pred", modelo_udf(...))

3. O DLT cuida sozinho de: orquestração, checkpoints, retries,
   qualidade de dados (expectations) e AUTOSCALING do compute
```

```python
# Esqueleto de uma pipeline DLT (roda dentro de uma DLT pipeline, não em notebook comum)
import dlt
import mlflow
from pyspark.sql import functions as F

MODEL_URI = "models:/workspace.default.churn_predictor@champion"

# spark_udf transforma o modelo MLflow em uma UDF Spark distribuída
predict_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double")

@dlt.table(name="eventos_brutos")
def eventos_brutos():
    return spark.readStream.table("workspace.default.eventos_clientes")

@dlt.table(name="eventos_pontuados")
@dlt.expect_or_drop("prob_valida", "prob_churn IS NOT NULL")
def eventos_pontuados():
    return (
        dlt.read_stream("eventos_brutos")
        .withColumn(
            "prob_churn",
            predict_udf(F.col("tenure_months"), F.col("monthly_charges"), F.col("total_charges")),
        )
    )
```

### Por que DLT em vez de um job de Structured Streaming

Esta é exatamente a distinção que a questão oficial testa:

| | **Structured Streaming (job próprio)** | **Delta Live Tables** |
|---|---|---|
| Você escreve | O loop, os checkpoints, o tratamento de erro | Só as transformações |
| Autoscaling do compute | Manual / limitado | **Automático e dinâmico** |
| Retries e recuperação de falha | Você implementa | Gerenciado |
| Qualidade de dados | Você implementa | `expectations` nativas |
| Linhagem entre tabelas | Você documenta | Automática |
| Orquestração de múltiplas tabelas | Você orquestra | Declarativa, por dependência |

> **A resposta oficial da Questão 5:** o cenário descreve dezenas de milhares de eventos por segundo, com volume flutuando ao longo do dia, exigindo que o compute **redimensione dinamicamente**. A resposta correta é **criar uma pipeline Delta Live Tables que aplica o algoritmo como uma Spark UDF** — porque o DLT faz autoscaling automático. As alternativas que chamam um *model serving endpoint* de dentro da pipeline estão erradas: um endpoint REST não aguenta dezenas de milhares de chamadas por segundo, e adiciona latência de rede a cada evento.

> **Regra geral:** para pontuar um fluxo de alto volume, **traga o modelo até os dados** (UDF distribuída). Chamar um endpoint REST por evento é o anti-padrão.

### Structured Streaming direto (sem DLT)

```python
# Alternativa mais simples, útil para entender o mecanismo
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd

# Pré-requisito: a tabela de origem precisa ter Change Data Feed habilitado
# spark.sql("ALTER TABLE workspace.default.churn_clientes "
#           "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

df_stream = spark.readStream.table("workspace.default.churn_clientes")

loaded_model = mlflow.sklearn.load_model(MODEL_URI)
features_cols = ["tenure_months", "monthly_charges", "total_charges"]

@pandas_udf(DoubleType())
def predict_stream_udf(*cols: pd.Series) -> pd.Series:
    df_f = pd.concat(cols, axis=1)
    df_f.columns = features_cols
    return pd.Series(loaded_model.predict_proba(df_f)[:, 1])

query = (df_stream
    .withColumn("prob_churn", predict_stream_udf(*[F.col(c) for c in features_cols]))
    .select("customer_id", "prob_churn")
    .writeStream
    .outputMode("append")
    # o checkpoint precisa de um Volume que EXISTA — crie antes:
    # CREATE VOLUME IF NOT EXISTS workspace.default.churn_project;
    .option("checkpointLocation", "/Volumes/workspace/default/churn_project/ckpt_stream")
    .toTable("workspace.default.churn_predictions_stream"))
```

> **O checkpoint é obrigatório** em qualquer `writeStream`. É ele que guarda o progresso e garante que, após uma falha, o processamento retoma de onde parou sem duplicar nem perder dados.

---

## 10.6 Modelo customizado (MLflow pyfunc)

Objetivo oficial: "fazer deploy de um modelo customizado em um endpoint". Um "modelo customizado" é aquele que embute lógica além do `predict()` do framework — regras de negócio, pré/pós-processamento, threshold próprio.

```python
import mlflow.pyfunc
import pandas as pd
import numpy as np

class ChurnModelWrapper(mlflow.pyfunc.PythonModel):
    """Modelo com threshold de negócio e ação recomendada."""

    def load_context(self, context):
        """Chamado UMA VEZ, ao carregar o modelo. Carregue artefatos aqui."""
        self.model = mlflow.sklearn.load_model(context.artifacts["modelo_base"])
        self.threshold = 0.4          # threshold de negócio, não o 0.5 padrão

    def predict(self, context, model_input, params=None):
        """OBRIGATÓRIO. model_input é um DataFrame pandas."""
        proba = self.model.predict_proba(model_input)[:, 1]
        return pd.DataFrame({
            "prob_churn": proba,
            "predicao":   (proba >= self.threshold).astype(int),
            "acao_recomendada": np.where(
                proba >= 0.7, "Contato urgente",
                np.where(proba >= 0.4, "Oferta de retencao", "Monitorar")
            ),
        })
```

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

# O artefato aponta para o modelo já registrado — sem pickle manual
with mlflow.start_run(run_name="churn_pyfunc_wrapper"):
    mlflow.pyfunc.log_model(
        name="model",
        python_model=ChurnModelWrapper(),
        artifacts={"modelo_base": MODEL_URI},
        input_example=pd.DataFrame({
            "tenure_months":   [2],
            "monthly_charges": [90.0],
            "total_charges":   [180.0],
        }),
        registered_model_name="workspace.default.churn_wrapper",   # sem hífen
    )

# Testar
wrapper = mlflow.pyfunc.load_model("models:/workspace.default.churn_wrapper/1")
resultado = wrapper.predict(pd.DataFrame({
    "tenure_months":   [2, 60, 15],
    "monthly_charges": [90.0, 45.0, 70.0],
    "total_charges":   [180.0, 2700.0, 1050.0],
}))
display(spark.createDataFrame(resultado))
```

Depois de registrado no Unity Catalog, esse modelo customizado vai para um endpoint **exatamente como qualquer outro** — Serving → Create endpoint → escolher `workspace.default.churn_wrapper`. Do ponto de vista do serving, um `pyfunc` customizado é indistinguível de um modelo sklearn.

| Método | Obrigatório? | Quando é chamado |
|---|---|---|
| `predict(self, context, model_input)` | ✅ **Sim** | A cada requisição |
| `load_context(self, context)` | Não | Uma vez, ao carregar o modelo |

> **Por que `load_context` existe:** carregar o modelo, abrir arquivos ou inicializar recursos dentro do `predict()` faria isso a **cada requisição**. O `load_context` roda uma vez só, na inicialização.

---

## 10.7 Dividir tráfego entre modelos (A/B testing)

Objetivo oficial: "dividir dados entre endpoints para inferência em tempo real". Na prática, isso é feito com **múltiplas served entities no mesmo endpoint**, cada uma recebendo uma porcentagem do tráfego.

```
        Requisições ao endpoint "churn-predictor-endpoint"
                          │
              ┌───────────┴───────────┐
              │                       │
           90% │                    10% │
              ▼                       ▼
      versão 1 (champion)      versão 2 (challenger)
```

```python
# Via SDK do Databricks
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route,
)

w = WorkspaceClient()

w.serving_endpoints.update_config(
    name="churn-predictor-endpoint",
    served_entities=[
        ServedEntityInput(
            name="churn-v1",
            entity_name="workspace.default.churn_predictor",
            entity_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True,
        ),
        ServedEntityInput(
            name="churn-v2",
            entity_name="workspace.default.churn_predictor",
            entity_version="2",
            workload_size="Small",
            scale_to_zero_enabled=True,
        ),
    ],
    traffic_config=TrafficConfig(routes=[
        Route(served_model_name="churn-v1", traffic_percentage=90),
        Route(served_model_name="churn-v2", traffic_percentage=10),
    ]),
)
```

Pela interface: **Serving → seu endpoint → Edit → Add served entity**, e ajuste o percentual de tráfego de cada uma.

### Os padrões que a prova nomeia

| Padrão | Como funciona |
|---|---|
| **A/B testing** | Tráfego dividido entre duas versões; compara-se o resultado de negócio de cada uma |
| **Canary deployment** | O novo modelo começa com uma fatia pequena (5–10%) e vai aumentando conforme se prova |
| **Shadow / dark launch** | O novo modelo recebe uma cópia do tráfego, mas a resposta dele é descartada — só se mede performance, sem risco |
| **Blue/green** | Dois ambientes completos; troca-se 100% do tráfego de uma vez, com rollback imediato |

> **Regras que caem:** a soma dos `traffic_percentage` precisa dar **100**. E múltiplas served entities convivem no **mesmo endpoint** — não é preciso criar um endpoint por versão.

> **Ligação com o Módulo 11:** o traffic split é o mecanismo para validar um **challenger** com tráfego real antes de promovê-lo a **champion**.

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| **Batch** | Alto volume, periódico, menor custo. Predição pronta antes da consulta |
| **Streaming** | Fluxo contínuo, latência de segundos, cluster ligado |
| **Real-time** | Endpoint REST, milissegundos, maior custo por predição |
| **Escolha** | Depende de *quando* a predição é necessária |
| `mlflow.pyfunc.load_model()` | Formato universal — qualquer framework |
| **Batch com pandas** | `toPandas()` só em amostras/datasets pequenos |
| `pandas_udf` | UDF vetorizada, opera em chunks pandas — a forma eficiente em escala |
| `mlflow.pyfunc.spark_udf()` | Transforma o modelo MLflow em UDF Spark distribuída |
| **Streaming com DLT** | Modelo como Spark UDF dentro de uma pipeline DLT; **autoscaling automático** |
| **Anti-padrão** | Chamar um endpoint REST por evento em stream de alto volume |
| `checkpointLocation` | Obrigatório em `writeStream` — garante retomada sem perda/duplicação |
| **Payload do endpoint** | `dataframe_records` ou `dataframe_split` |
| **Scale to zero** | Zera o custo sem tráfego, ao preço do cold start |
| `PythonModel` | `predict()` é **obrigatório**; `load_context()` é opcional e roda uma vez |
| **Modelo customizado** | É registrado e servido igual a qualquer outro modelo do UC |
| **Traffic split** | Múltiplas served entities no **mesmo** endpoint, percentuais somando **100** |
| **Canary / A-B / shadow / blue-green** | Estratégias de rollout progressivo |

---

→ Próximo: [11_mlops.md](11_mlops.md)
