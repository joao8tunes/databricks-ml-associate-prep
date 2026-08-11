# Módulo 9 — Feature Store (Feature Engineering no Unity Catalog)

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 1 — Databricks Machine Learning (**38% da prova**)
>
> **Notebook sugerido:** `09_feature_store`
>
> **Pré-requisito:** Módulos 2 (tabela `churn_clientes`) e 4 (MLflow).

Cinco objetivos oficiais da prova estão neste módulo — é o segundo tema mais denso depois do MLflow. E tem uma armadilha: **a maior parte do material antigo na internet ensina a API errada.**

**Objetivos oficiais cobertos neste módulo:**
- Identificar os benefícios de criar feature tables no nível de conta (Unity Catalog) vs no nível de workspace
- Criar uma feature table no Unity Catalog
- Escrever dados em uma feature table
- Treinar um modelo com features de uma feature table
- Pontuar um modelo usando features de uma feature table
- Descrever as diferenças entre feature tables online e offline

---

## 9.1 O problema que o Feature Store resolve

```
Sem Feature Store:
├── Time A calcula "receita_media_90_dias" de um jeito (para o treino)
├── Time B calcula a mesma feature de outro jeito (para produção)
├── Divergência silenciosa → o modelo performa pior em produção do que no teste
└── Cada projeto recalcula as mesmas features do zero

Com Feature Store:
├── Feature calculada uma vez, armazenada centralmente
├── Reutilizada em múltiplos projetos e modelos
├── O MESMO cálculo alimenta treino e inferência
├── Point-in-time lookups evitam data leakage temporal
└── Linhagem e documentação automáticas
```

> **O termo que a prova usa:** o problema resolvido é o **training-serving skew** — a divergência entre as features vistas no treino e as vistas em produção.

---

## 9.2 Unity Catalog vs Workspace Feature Store

Existem **duas gerações** de Feature Store no Databricks, com clients Python diferentes. Confundir as duas é o erro mais comum.

| | **Workspace Feature Store** (legado) | **Feature Engineering no Unity Catalog** (atual) |
|---|---|---|
| **Client Python** | `FeatureStoreClient` | **`FeatureEngineeringClient`** |
| **Pacote** | `databricks.feature_store` | **`databricks.feature_engineering`** |
| **Nome da tabela** | `nome_da_tabela` (1 nível) | **`catalog.schema.tabela`** (3 níveis) |
| **Escopo** | Um workspace | **A conta inteira** (metastore) |
| **O que é por baixo** | Formato proprietário | **Uma tabela Delta comum com primary key** |
| **Status** | Depreciado | **Recomendado** |

> ⚠️ **`databricks-feature-store` foi depreciado a partir da versão 0.17.0** e seus módulos migraram para `databricks-feature-engineering`. Em qualquer workspace com Unity Catalog — o que inclui a Free Edition — use **`FeatureEngineeringClient`**.

### Os benefícios de criar feature tables no Unity Catalog (objetivo oficial)

| Benefício | Detalhe |
|---|---|
| **Escopo de conta, não de workspace** | A feature table fica no metastore e é visível de **todos os workspaces** da conta. No modelo antigo, ela morria dentro de um workspace |
| **Governança granular** | `GRANT SELECT` por tabela/schema/catalog, integrado ao mesmo controle de acesso dos dados |
| **É uma tabela Delta comum** | Qualquer ferramenta que lê Delta lê a feature table. Sem formato proprietário. Você ganha time travel, ACID e schema enforcement de graça |
| **Linhagem automática** | O Unity Catalog rastreia quais tabelas originaram a feature e quais modelos a consomem |
| **Namespace único** | Mesmo padrão `catalog.schema.objeto` de tabelas, volumes e modelos |
| **Descoberta** | Times diferentes encontram e reutilizam features já calculadas, em vez de recriá-las |

> **Esta é literalmente a Questão 1 do exam guide oficial.** O enunciado descreve um workspace com Unity Catalog habilitado e pergunta como criar a feature table. A resposta correta é: **usar o método `create_table` do `FeatureEngineeringClient` em Python e depois escrever os dados**. Não existe cláusula SQL `AS FEATURE STORE` nem `ALTER TABLE ... SET AS FEATURE STORE` — essas alternativas são inventadas.

---

## 9.3 Criar uma feature table no Unity Catalog

```python
# Se der ImportError, instale e reinicie o Python do notebook:
# %pip install databricks-feature-engineering
# dbutils.library.restartPython()

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()

# Tabela criada no Módulo 2 — rode aquele notebook primeiro se ainda não rodou
df = spark.table("workspace.default.churn_clientes")

# --- 1. Centralizar o cálculo das features em uma função ---
# Esta função é a "fonte da verdade": treino e produção chamam a MESMA lógica
def calcular_features_cliente(df):
    return df.select(
        "customer_id",                                  # PRIMARY KEY — obrigatória
        F.col("tenure_months").cast("double").alias("tenure_months"),
        F.col("monthly_charges"),
        F.col("total_charges"),
        # Features derivadas
        (F.col("total_charges") / F.col("tenure_months")).alias("avg_monthly_spend"),
        F.when(F.col("monthly_charges") >= 80, 1.0).otherwise(0.0).alias("is_premium"),
        F.when(F.col("contract_type") == "Month-to-month", 1.0).otherwise(0.0).alias("is_monthly_contract"),
        F.when(F.col("internet_service") == "Fiber optic", 1.0).otherwise(0.0).alias("has_fiber"),
        F.when(F.col("tech_support") == "Yes", 1.0).otherwise(0.0).alias("has_tech_support"),
    )

features_df = calcular_features_cliente(df)
display(features_df.limit(5))
```

```python
FEATURE_TABLE = "workspace.default.features_clientes_churn"   # catalog.schema.tabela

# --- 2. Criar a tabela de features (uma única vez) ---
# Duas formas equivalentes:

# Forma A: criar já com os dados
fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["customer_id"],     # obrigatório — é a chave de lookup
    df=features_df,                   # cria a tabela E escreve os dados
    description="Features de clientes para o modelo de churn. Atualizada diariamente.",
)

# Forma B: criar vazia a partir do schema, e escrever depois
# fe.create_table(
#     name=FEATURE_TABLE,
#     primary_keys=["customer_id"],
#     schema=features_df.schema,      # só o schema, sem dados
#     description="...",
# )
# fe.write_table(name=FEATURE_TABLE, df=features_df, mode="merge")

print("Feature table criada!")
```

> **O que torna uma tabela Delta em feature table:** a **primary key**. Por baixo dos panos, `create_table` cria uma tabela Delta comum com uma constraint de chave primária. Não existe um formato especial.

---

## 9.4 Escrever e ler dados da feature table

```python
# --- Atualizar features (a rotina do dia a dia) ---
fe.write_table(
    name=FEATURE_TABLE,
    df=features_df,
    mode="merge",     # "merge": upsert — atualiza pela PK e insere as novas
                      # "overwrite": substitui a tabela inteira
)
print("Features atualizadas!")

# --- Ler a feature table como DataFrame Spark ---
features_lidas = fe.read_table(name=FEATURE_TABLE)
display(features_lidas.limit(5))
print(f"Features (sem contar a PK): {len(features_lidas.columns) - 1}")

# --- Metadados da tabela ---
meta = fe.get_table(name=FEATURE_TABLE)
print(f"Nome        : {meta.name}")
print(f"Descrição   : {meta.description}")
print(f"Primary keys: {meta.primary_keys}")
```

| `mode` | Comportamento |
|---|---|
| `"merge"` | **Upsert**: atualiza as linhas cuja primary key já existe e insere as novas. É o modo do dia a dia |
| `"overwrite"` | Substitui **toda** a tabela pelos dados fornecidos |

---

## 9.5 Criar o training set com FeatureLookup

A ideia central: seu DataFrame de treino contém **apenas as chaves e o label**. As features são buscadas automaticamente.

```python
from databricks.feature_engineering import FeatureLookup

# O DataFrame de labels tem só customer_id + churn
df_labels = df.select("customer_id", "churn")

# FeatureLookup = "busque estas features, desta tabela, por esta chave"
feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        feature_names=[
            "tenure_months",
            "monthly_charges",
            "avg_monthly_spend",
            "is_premium",
            "is_monthly_contract",
            "has_fiber",
            "has_tech_support",
        ],
        lookup_key="customer_id",      # coluna do df_labels que faz o join
    )
]

training_set = fe.create_training_set(
    df=df_labels,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id"],   # o ID não deve virar feature
)

training_df = training_set.load_df()
display(training_df.limit(5))
print("Colunas:", training_df.columns)
```

> **Por que não fazer um join manual?** Porque o valor não é o join — é o **rastreamento**. O objeto `training_set` registra quais feature tables foram usadas, quais colunas e por qual chave. Isso é o que habilita a reprodutibilidade e o `score_batch` automático da seção 9.7.

---

## 9.6 Treinar e logar o modelo vinculado à Feature Store

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_registry_uri("databricks-uc")

df_pd = training_df.toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# USE fe.log_model(), NÃO mlflow.sklearn.log_model()
# fe.log_model() grava, junto com o modelo, os metadados de quais features
# buscar e como — é isso que habilita o score_batch automático
MODEL_NAME = "workspace.default.churn_fs_model"   # nome UC: sem hífen

with mlflow.start_run(run_name="churn_feature_store"):
    mlflow.log_metric("auc_roc", auc)

    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,            # ← o vínculo com a Feature Store
        registered_model_name=MODEL_NAME,
    )

print(f"AUC-ROC: {auc:.4f}")
print("Modelo registrado e vinculado à Feature Store!")
```

| | `mlflow.sklearn.log_model()` | `fe.log_model()` |
|---|---|---|
| Salva o modelo | ✅ | ✅ |
| Salva **quais features buscar e onde** | ❌ | ✅ |
| Habilita `fe.score_batch()` sem código de join | ❌ | ✅ |
| Registra linhagem modelo → feature table | ❌ | ✅ |

---

## 9.7 Scoring em batch com `score_batch`

```python
from mlflow.tracking import MlflowClient

# Apontar o alias "champion" para a versão mais recente do modelo
client = MlflowClient()

ultima_versao = max(
    int(mv.version) for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
)
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=ultima_versao)
print(f"Alias 'champion' → versão {ultima_versao}")

# --- Pontuar clientes passando SÓ a chave ---
# Nenhuma feature no DataFrame de entrada. O Feature Store busca tudo.
novos_clientes = spark.createDataFrame(
    [("C0001",), ("C0002",), ("C0100",), ("C0200",)],
    ["customer_id"],
)

predictions = fe.score_batch(
    model_uri=f"models:/{MODEL_NAME}@champion",
    df=novos_clientes,
    result_type="double",
)

display(predictions.select("customer_id", "prediction"))
```

> **O ponto da prova:** você passou um DataFrame com **apenas `customer_id`** e recebeu predições. O `fe.score_batch()` leu os metadados gravados pelo `fe.log_model()`, descobriu quais feature tables consultar, fez o lookup e aplicou o modelo. Se você tivesse usado `mlflow.sklearn.log_model()`, isso não funcionaria.

---

## 9.8 Online vs offline feature tables

Objetivo oficial: "descrever as diferenças entre feature tables online e offline". É conceitual — a Free Edition não permite criar online stores.

```
OFFLINE (o padrão, o que você criou acima)
├── Armazenamento: tabela Delta no Unity Catalog
├── Otimizada para: LEITURA EM VOLUME (milhões de linhas de uma vez)
├── Latência: segundos a minutos
├── Usada em: TREINO e BATCH INFERENCE
└── Guarda: histórico completo das features

ONLINE (feature serving)
├── Armazenamento: banco de baixa latência (Databricks Online Tables / Online Feature Store)
├── Otimizada para: LEITURA PONTUAL por chave (um cliente por vez)
├── Latência: milissegundos
├── Usada em: REAL-TIME SERVING (endpoint REST)
└── Guarda: apenas o valor MAIS RECENTE de cada chave
```

| | **Offline** | **Online** |
|---|---|---|
| **Onde vive** | Delta / Unity Catalog | Store de baixa latência |
| **Padrão de acesso** | Varredura em lote | Lookup por chave |
| **Latência** | Segundos–minutos | Milissegundos |
| **Volume por consulta** | Milhões de linhas | Uma ou poucas linhas |
| **Caso de uso** | Treino, batch scoring | Inferência em tempo real |
| **Histórico** | Completo | Só o valor atual |
| **Como é populada** | Job que calcula e grava | **Sincronizada a partir da offline** |

> **A relação entre as duas:** a tabela offline é a fonte da verdade. A online é uma **cópia sincronizada** dela, contendo só o valor mais recente, publicada com `fe.publish_table(...)`. O mesmo cálculo alimenta as duas — é assim que o Feature Store elimina o training-serving skew mesmo em tempo real.

> ⚠️ **Free Edition:** online tables constam explicitamente na lista de recursos não suportados. Estude o conceito; não tente criar.

---

## 9.9 Point-in-time lookups

> ⚠️ Código ilustrativo — `features_historico_clientes` é uma tabela hipotética que não foi criada nas seções acima. Não precisa rodar; é para entender o conceito.

```python
# Point-in-time lookup evita data leakage TEMPORAL:
# "qual era o valor da feature NA DATA do evento?" — não o valor de hoje.

# PROBLEMA sem point-in-time:
# Você prevê churn de um cliente em Janeiro.
# Sem point-in-time, a feature "total_charges" pegaria o valor de hoje (Dezembro),
# que embute informação do futuro em relação ao evento de Janeiro.
# O modelo fica ótimo na validação e péssimo em produção.

from databricks.feature_engineering import FeatureLookup
from pyspark.sql import functions as F

df_eventos = df_labels.withColumn("event_ts", F.to_timestamp(F.lit("2024-01-01")))

feature_lookups_temporais = [
    FeatureLookup(
        table_name="workspace.default.features_historico_clientes",
        feature_names=["avg_monthly_spend", "is_premium"],
        lookup_key="customer_id",
        timestamp_lookup_key="event_ts",   # ← busca o valor vigente NESTA data
    )
]
```

Para isso funcionar, a feature table precisa ser criada como **time series feature table**, declarando a coluna de tempo:

```python
# fe.create_table(
#     name="workspace.default.features_historico_clientes",
#     primary_keys=["customer_id", "feature_ts"],
#     timeseries_columns="feature_ts",     # ← declara a coluna temporal
#     df=features_historicas_df,
# )
```

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| **`FeatureEngineeringClient`** | O client atual, de `databricks.feature_engineering`. `FeatureStoreClient` é o legado |
| **Nome de 3 níveis** | `catalog.schema.tabela` — obrigatório no Unity Catalog |
| **`primary_keys`** | Obrigatória na criação. É o que transforma uma tabela Delta em feature table |
| **`fe.create_table()`** | Cria a feature table. Aceita `df=` (com dados) ou `schema=` (vazia) |
| **`fe.write_table(mode=...)`** | `"merge"` = upsert pela PK; `"overwrite"` = substitui tudo |
| **`FeatureLookup`** | Diz quais features buscar, de qual tabela e por qual `lookup_key` |
| **`fe.create_training_set()`** | Vincula labels + features, gerando linhagem rastreável |
| **`exclude_columns`** | Remove colunas que não devem virar feature (IDs, timestamps) |
| **`fe.log_model()`** | Loga o modelo **com** os metadados de lookup — é o que habilita `score_batch` |
| **`fe.score_batch()`** | Recebe só as chaves, busca as features e prediz |
| **`timestamp_lookup_key`** | Point-in-time lookup — evita leakage temporal |
| **Offline** | Delta/UC, leitura em volume, treino e batch, histórico completo |
| **Online** | Baixa latência, lookup por chave, real-time serving, só o valor atual |
| **Training-serving skew** | O problema que o Feature Store existe para resolver |
| **UC vs workspace** | Escopo de conta, governança granular, tabela Delta comum, linhagem automática |

---

→ Próximo: [10_deployment.md](10_deployment.md)
