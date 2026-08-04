# Módulo 8 — Feature Store

> **Notebook sugerido:** `08_feature_store`
>
> Feature Store é o repositório central de features reutilizáveis entre projetos.
> Garante que treino e produção usam exatamente as mesmas features.

---

## 8.1 O problema que o Feature Store resolve

```
Sem Feature Store:
├── Time A calcula "receita_media_90_dias" de um jeito (para treino)
├── Time B calcula a mesma feature de outro jeito (para produção)
├── Divergência silenciosa → modelo performa mal em produção
└── Cada projeto recalcula as mesmas features do zero

Com Feature Store:
├── Feature calculada uma vez, armazenada centralmente
├── Reutilizada em múltiplos projetos e modelos
├── Mesmo cálculo garante consistência treino ↔ produção
├── Point-in-time lookups evitam data leakage temporal
└── Documentação e rastreabilidade automáticas
```

---

## 8.2 Criar e escrever features

```python
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import functions as F

fs = FeatureStoreClient()

df = spark.table("workspace.default.churn_clientes")

# --- 1. Calcular as features (lógica centralizada) ---
def calcular_features_cliente(df):
    return df.select(
        "customer_id",                     # PRIMARY KEY — obrigatória!
        F.col("tenure_months").cast("double"),
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

# --- 2. Criar a tabela de features (primeira vez) ---
fs.create_table(
    name="workspace.default.features_clientes_churn",   # "catalog.schema.table_name"
    primary_keys=["customer_id"],              # chave para fazer lookup
    df=features_df,
    description="Features de clientes para o modelo de churn. Atualizada diariamente."
)
print("Feature table criada!")

# --- 3. Atualizar features (dia a dia) ---
fs.write_table(
    name="workspace.default.features_clientes_churn",
    df=features_df,
    mode="merge"     # "merge": atualiza existentes + insere novos
                     # "overwrite": substitui toda a tabela
)
print("Features atualizadas!")
```

---

## 8.3 Ler features diretamente

```python
# Ler a tabela de features como DataFrame Spark
features_lidas = fs.read_table("workspace.default.features_clientes_churn")
display(features_lidas.limit(5))
print(f"Total de features: {len(features_lidas.columns) - 1}")  # -1 para não contar a PK
```

---

## 8.4 Criar training set com FeatureLookup

```python
from databricks.feature_store import FeatureLookup

# df_labels contém APENAS customer_id + label (churn)
# As features são buscadas automaticamente pelo Feature Store
df_labels = df.select("customer_id", "churn")

# FeatureLookup = instrução de "buscar estas features pelo customer_id"
feature_lookups = [
    FeatureLookup(
        table_name="workspace.default.features_clientes_churn",
        feature_names=[
            "tenure_months",
            "monthly_charges",
            "avg_monthly_spend",
            "is_premium",
            "is_monthly_contract",
            "has_fiber",
            "has_tech_support",
        ],
        lookup_key="customer_id"    # chave de join
    )
]

# Criar o training set (vincula features ao conjunto de treino)
training_set = fs.create_training_set(
    df=df_labels,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id"]   # não usar ID como feature
)

# Carregar como DataFrame Spark (com todas as features)
training_df = training_set.load_df()
display(training_df.limit(5))
print(f"Colunas: {training_df.columns}")
```

---

## 8.5 Treinar e logar modelo vinculado ao Feature Store

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow

# Converter para pandas para sklearn
df_pd = training_df.toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# USAR fs.log_model() em vez de mlflow.sklearn.log_model()
# fs.log_model() vincula o modelo ao Feature Store → habilita score_batch automático!
with mlflow.start_run(run_name="churn_feature_store"):
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("auc_roc", auc)

    fs.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,               # ← vínculo com o Feature Store
        registered_model_name="churn-fs-model"   # registra no Registry automaticamente
    )

print(f"AUC-ROC: {auc:.4f}")
print("Modelo registrado e vinculado ao Feature Store!")
```

---

## 8.6 Scoring em batch com Feature Store

```python
# Clientes para pontuar — só precisa do customer_id!
# O Feature Store busca as features automaticamente
novos_clientes = spark.createDataFrame(
    [("C0001",), ("C0002",), ("C0100",), ("C0200",)],
    ["customer_id"]
)

# score_batch: busca as features, aplica o modelo e retorna predições
predictions = fs.score_batch(
    model_uri="models:/churn-fs-model/Production",
    df=novos_clientes,
    result_type="double"    # tipo do retorno: "double", "int", "string", etc.
)

display(predictions.select("customer_id", "prediction"))
```

---

## 8.7 Point-in-time lookups (conceito da prova)

```python
# Point-in-time lookup evita data leakage em dados temporais:
# "Qual era o valor da feature na data X?" (não o valor atual)

# PROBLEMA sem point-in-time:
# Você quer prever churn de cliente em Janeiro.
# Sem point-in-time, a feature "total_charges" poderia usar o valor de Dezembro
# (que inclui informações do futuro em relação ao evento de Janeiro).

# COM point-in-time lookup:
df_eventos = df_labels.withColumn("timestamp", F.to_timestamp(F.lit("2024-01-01")))

feature_lookups_temporais = [
    FeatureLookup(
        table_name="workspace.default.features_historico_clientes",
        feature_names=["avg_monthly_spend", "is_premium"],
        lookup_key="customer_id",
        timestamp_lookup_key="timestamp"  # ← busca o valor da feature NESTA data
    )
]
# Garante que você usa apenas informações disponíveis até aquela data
```

---

## 8.8 Listar e explorar Feature Tables

```python
# Listar tabelas de features
client_fs = FeatureStoreClient()

# Obter metadados de uma tabela
table_meta = client_fs.get_table("workspace.default.features_clientes_churn")
print(f"Nome: {table_meta.name}")
print(f"Descrição: {table_meta.description}")
print(f"Primary keys: {table_meta.primary_keys}")

# Procurar features por nome ou descrição
# (via UI: Machine Learning → Feature Store → Browse tables)
```

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| `primary_keys` | Chave usada para fazer lookup (join) — obrigatória na criação |
| `FeatureLookup` | Especifica quais features buscar, de qual tabela e por qual chave |
| `mode="merge"` | Atualiza existentes + insere novos |
| `mode="overwrite"` | Substitui toda a tabela |
| `fs.log_model()` | **Vincula** o modelo ao Feature Store (vs `mlflow.sklearn.log_model`) |
| `fs.score_batch()` | Busca features automaticamente e aplica o modelo em lote |
| `training_set` | Objeto que vincula features + labels para treino rastreável |
| `timestamp_lookup_key` | Para point-in-time lookup (evitar data leakage temporal) |
| `exclude_columns` | Remove colunas que não devem ser features (ex: IDs, timestamps) |
| Training-serving skew | O principal problema que o Feature Store resolve |

---

→ Próximo: [09_deployment.md](09_deployment.md)
