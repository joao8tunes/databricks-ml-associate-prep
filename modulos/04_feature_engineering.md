# Módulo 4 — Feature Engineering

> **Notebook sugerido:** `04_feature_engineering`
>
> Spark ML usa o mesmo padrão do sklearn: `fit()` aprende parâmetros, `transform()` aplica.
> A grande diferença: tudo opera sobre DataFrames Spark distribuídos.

---

## 4.1 Transformadores essenciais — o que cai na prova

| Transformador | Entrada | Saída | Uso |
|---|---|---|---|
| `StringIndexer` | coluna texto | coluna numérica (índice) | Label encoding |
| `OneHotEncoder` | coluna de índices | coluna de vetor binário | OHE |
| `VectorAssembler` | múltiplas colunas | uma coluna Vector | Montar features |
| `StandardScaler` | coluna Vector | coluna Vector normalizada | Normalização (média 0, std 1) |
| `MinMaxScaler` | coluna Vector | coluna Vector em [0,1] | Escala min-max |
| `Imputer` | colunas numéricas | colunas sem nulos | Preencher nulos |
| `Bucketizer` | coluna numérica | coluna categórica | Bins com limites fixos |
| `QuantileDiscretizer` | coluna numérica | coluna categórica | Bins por quantis |
| `PCA` | coluna Vector | coluna Vector menor | Redução de dimensionalidade |

```python
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, MinMaxScaler, Imputer,
    Bucketizer, QuantileDiscretizer, PCA,
)
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
```

---

## 4.2 Pipeline completo — do DataFrame bruto às features prontas

```python
df = spark.read.format("delta").load("dbfs:/FileStore/churn_project/data/clientes")

# Injetar alguns nulos para praticar Imputer
import random
random.seed(0)
ids_nulos = [f"C{i:04d}" for i in random.sample(range(1, 1001), 50)]
df = df.withColumn(
    "monthly_charges",
    F.when(F.col("customer_id").isin(ids_nulos), None).otherwise(F.col("monthly_charges"))
)

# Identificar colunas por tipo
colunas_num = ["tenure_months", "monthly_charges", "total_charges"]
colunas_cat = ["contract_type", "internet_service", "tech_support", "payment_method"]

# --- PASSO 1: Preencher nulos ---
imputer = Imputer(
    inputCols=colunas_num,
    outputCols=[c + "_imp" for c in colunas_num],
    strategy="median"   # "mean" ou "mode" também disponíveis
)

# --- PASSO 2: Label encoding (texto → índice numérico) ---
# StringIndexer PRECISA vir antes do OneHotEncoder
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"   # "error" (default) | "skip" | "keep"
    )
    for c in colunas_cat
]

# --- PASSO 3: One-Hot Encoding ---
encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_ohe",
        dropLast=True   # default True: evita multicolinearidade
    )
    for c in colunas_cat
]

# --- PASSO 4: Juntar tudo em um único vetor "features" ---
# OBRIGATÓRIO: modelos Spark ML exigem uma coluna Vector chamada "features"
assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in colunas_num] + [c + "_ohe" for c in colunas_cat],
    outputCol="features_raw",
    handleInvalid="skip"    # "error" (default) | "skip" | "keep"
)

# --- PASSO 5: Normalizar ---
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,   # centraliza em zero (subtrai média)
    withStd=True     # escala pelo desvio padrão
)

# --- PASSO 6: Montar e treinar o Pipeline ---
pipeline = Pipeline(stages=[imputer] + indexers + encoders + [assembler, scaler])

# IMPORTANTE: split ANTES de fit para evitar data leakage
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# fit() aprende parâmetros APENAS no treino
pipeline_model = pipeline.fit(train_df)

# transform() aplica os parâmetros aprendidos
train_feat = pipeline_model.transform(train_df)
test_feat  = pipeline_model.transform(test_df)

display(train_feat.select("customer_id", "features", "churn").limit(5))
print(f"Tamanho do vetor de features: {len(train_feat.select('features').first()[0])}")
```

---

## 4.3 Bucketizer — bins com limites fixos

```python
from pyspark.ml.feature import Bucketizer

# Dividir tenure_months em faixas fixas
bucketizer = Bucketizer(
    splits=[0, 12, 24, 48, float("inf")],  # cria 4 buckets: [0-12), [12-24), [24-48), [48-∞)
    inputCol="tenure_months",
    outputCol="tenure_bucket",
    handleInvalid="keep"
)

df_bucket = bucketizer.transform(df)
display(
    df_bucket.groupBy("tenure_bucket").agg(
        F.count("*").alias("clientes"),
        F.round(F.mean("churn") * 100, 1).alias("taxa_churn_%")
    ).orderBy("tenure_bucket")
)
```

---

## 4.4 QuantileDiscretizer — bins automáticos por quantis

```python
from pyspark.ml.feature import QuantileDiscretizer

# Divide automaticamente em quartis (sem precisar definir os limites)
discretizer = QuantileDiscretizer(
    numBuckets=4,
    inputCol="monthly_charges",
    outputCol="charges_quartile"
)

# Diferença do Bucketizer:
# Bucketizer    → você define os limites manualmente
# QuantileDiscretizer → Spark calcula os limites para gerar grupos de tamanho igual

df_disc = discretizer.fit(df).transform(df)
display(
    df_disc.groupBy("charges_quartile")
           .agg(F.count("*").alias("clientes"))
           .orderBy("charges_quartile")
)
```

---

## 4.5 StringIndexer com múltiplas colunas

```python
# Indexar várias colunas de uma vez
multi_indexer = StringIndexer(
    inputCols=["contract_type", "internet_service", "payment_method"],
    outputCols=["contract_idx", "internet_idx", "payment_idx"],
    handleInvalid="keep"
)

df_indexed = multi_indexer.fit(df).transform(df)
display(df_indexed.select("contract_type", "contract_idx", "internet_service", "internet_idx").limit(5))
```

---

## 4.6 PCA — redução de dimensionalidade

```python
from pyspark.ml.feature import PCA

# PCA reduz a dimensão do vetor de features (precisa de um Vector como input)
pca = PCA(k=2, inputCol="features_raw", outputCol="features_pca")
pca_model = pca.fit(train_feat)

df_pca = pca_model.transform(train_feat)

# Variância explicada por cada componente
print("Variância explicada por componente:")
for i, v in enumerate(pca_model.explainedVariance):
    print(f"  PC{i+1}: {v:.2%}")
print(f"Total: {sum(pca_model.explainedVariance):.2%}")
```

---

## 4.7 Por que a ORDEM importa no Pipeline

```python
# ERRADO: OneHotEncoder antes do StringIndexer
pipeline_errado = Pipeline(stages=[
    OneHotEncoder(inputCol="contract_type", outputCol="contract_ohe"),  # ← ERRO: OHE precisa de índice
    StringIndexer(inputCol="contract_type", outputCol="contract_idx"),
])

# CORRETO: StringIndexer sempre primeiro
pipeline_correto = Pipeline(stages=[
    StringIndexer(inputCol="contract_type", outputCol="contract_idx"),   # texto → número
    OneHotEncoder(inputCol="contract_idx", outputCol="contract_ohe"),     # número → vetor binário
    VectorAssembler(inputCols=["contract_ohe", "tenure_months"], outputCol="features"),
])

# ERRADO: fit no dataset completo (data leakage!)
pipeline_model_errado = pipeline_correto.fit(df)  # ← usa dados de teste para aprender stats

# CORRETO: fit apenas no treino
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
pipeline_model_correto = pipeline_correto.fit(train_df)  # ← só aprende com o treino
train_result = pipeline_model_correto.transform(train_df)
test_result  = pipeline_model_correto.transform(test_df)
```

---

## 4.8 MinMaxScaler

```python
from pyspark.ml.feature import MinMaxScaler

# MinMaxScaler: escala para [0, 1] (ou range customizado)
# Diferente do StandardScaler (que centraliza em zero)
min_max = MinMaxScaler(
    inputCol="features_raw",
    outputCol="features_minmax",
    min=0.0,
    max=1.0
)

# Quando usar qual?
# StandardScaler → quando features seguem distribuição normal, algoritmos sensíveis à escala
# MinMaxScaler   → quando quer valores no range [0,1], redes neurais
```

---

## Exercício Prático

```python
# Construir um pipeline completo do zero e comparar com e sem normalização:

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df = spark.read.format("delta").load("dbfs:/FileStore/churn_project/data/clientes")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Definir stages
imputer   = Imputer(inputCols=["tenure_months", "monthly_charges", "total_charges"],
                    outputCols=["tenure_imp", "monthly_imp", "total_imp"])
indexer   = StringIndexer(inputCol="contract_type", outputCol="contract_idx", handleInvalid="keep")
encoder   = OneHotEncoder(inputCol="contract_idx", outputCol="contract_ohe")
assembler = VectorAssembler(inputCols=["tenure_imp", "monthly_imp", "total_imp", "contract_ohe"],
                             outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features")
lr        = LogisticRegression(featuresCol="features", labelCol="churn")

pipeline  = Pipeline(stages=[imputer, indexer, encoder, assembler, scaler, lr])

# Treinar o pipeline inteiro de uma vez!
model     = pipeline.fit(train_df)
preds     = model.transform(test_df)

evaluator = BinaryClassificationEvaluator(labelCol="churn")
auc       = evaluator.evaluate(preds)
print(f"AUC-ROC do pipeline completo: {auc:.4f}")
```

---

## Pontos-chave para a prova

- Ordem obrigatória: `StringIndexer → OneHotEncoder → VectorAssembler`
- VectorAssembler cria uma coluna do tipo `Vector` (obrigatória para modelos Spark ML)
- `handleInvalid="keep"` → cria categoria extra para valores desconhecidos
- `handleInvalid="skip"` → remove linhas com valores desconhecidos
- `Pipeline.fit()` retorna um `PipelineModel` (objeto com os stages treinados)
- **Data leakage:** sempre fit no treino, transform no treino e no teste separadamente
- `Bucketizer` → limites fixos definidos por você
- `QuantileDiscretizer` → limites automáticos (quantis iguais)
- `StandardScaler(withMean=True)` → subtrai média (centraliza em zero)
- `MinMaxScaler` → escala para [0, 1]

---

→ Próximo: [05_treinamento_modelos.md](05_treinamento_modelos.md)
