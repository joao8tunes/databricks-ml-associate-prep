# Módulo 5 — Feature Engineering com Spark ML

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 3 — Model Development (**31% da prova**)
>
> **Notebook sugerido:** `05_feature_engineering`
>
> **Pré-requisito:** Módulos 2 e 3.

Spark ML usa o mesmo padrão do scikit-learn: `fit()` aprende, `transform()` aplica. A diferença é que tudo opera sobre DataFrames Spark distribuídos.

**Objetivos oficiais cobertos neste módulo:**
- Comparar estimators e transformers
- Desenvolver uma training pipeline
- Usar one-hot encoding para features categóricas

---

## 5.1 Estimators vs Transformers

Este é um objetivo oficial e uma distinção que a prova cobra diretamente.

```
TRANSFORMER
├── Tem: transform()
├── NÃO aprende nada dos dados
├── Já sabe o que fazer no momento em que é criado
└── DataFrame → DataFrame

ESTIMATOR
├── Tem: fit()
├── APRENDE parâmetros a partir dos dados
├── fit() devolve um Transformer (o "Model")
└── DataFrame → Transformer
```

```
     Estimator          .fit(dados)          Transformer/Model
  ┌──────────────┐                        ┌──────────────────┐
  │ StringIndexer│ ────────────────────→  │StringIndexerModel│
  │ (não sabe as │   aprende a lista      │ (sabe: Fibra=0,  │
  │  categorias) │   de categorias        │  DSL=1, No=2)    │
  └──────────────┘                        └──────────────────┘
                                                   │
                                            .transform(dados)
                                                   ↓
                                              DataFrame
```

| Componente | Tipo | Por quê |
|---|---|---|
| `StringIndexer` | **Estimator** | Precisa varrer os dados para descobrir as categorias e sua frequência |
| `StringIndexerModel` | Transformer | Já tem o mapeamento categoria → índice |
| `OneHotEncoder` | **Estimator** | Precisa saber quantas categorias existem para dimensionar o vetor |
| `Imputer` | **Estimator** | Precisa calcular média/mediana/moda |
| `StandardScaler` | **Estimator** | Precisa calcular média e desvio padrão |
| `MinMaxScaler` | **Estimator** | Precisa calcular mínimo e máximo |
| `QuantileDiscretizer` | **Estimator** | Precisa calcular os quantis |
| `PCA` | **Estimator** | Precisa calcular os componentes principais |
| `VectorAssembler` | **Transformer** | Só concatena colunas — não aprende nada |
| `Bucketizer` | **Transformer** | Você já forneceu os `splits` — não aprende nada |
| `SQLTransformer` | **Transformer** | Só executa a query que você escreveu |
| `Tokenizer` | **Transformer** | Só quebra strings |
| `RandomForestClassifier` | **Estimator** | Treina o modelo |
| `RandomForestClassificationModel` | Transformer | Já treinado, só prediz |
| `Pipeline` | **Estimator** | `fit()` devolve um `PipelineModel` |
| `PipelineModel` | Transformer | Todos os stages já ajustados |

> **A regra que resolve qualquer questão:** *se o componente precisa olhar os dados para aprender alguma coisa, é um **Estimator**; se ele já nasce sabendo o que fazer, é um **Transformer**.*

> **Pegadinhas frequentes:** `VectorAssembler` e `Bucketizer` são **Transformers** (não têm `fit()`), enquanto `OneHotEncoder` no Spark **é um Estimator** (diferente do sklearn, onde a intuição de muita gente é de que "só codifica"). E todo **modelo** treinado é um Transformer — `transform()` é o que gera as predições.

---

## 5.2 Transformadores essenciais

| Componente | Entrada | Saída | Uso | Tipo |
|---|---|---|---|---|
| `StringIndexer` | coluna texto | índice numérico | Label encoding | Estimator |
| `OneHotEncoder` | coluna de índices | vetor binário | OHE | Estimator |
| `VectorAssembler` | várias colunas | uma coluna Vector | Montar `features` | Transformer |
| `StandardScaler` | Vector | Vector padronizado | Média 0, desvio 1 | Estimator |
| `MinMaxScaler` | Vector | Vector em [0,1] | Escala min-max | Estimator |
| `Imputer` | colunas numéricas | colunas sem nulos | Preencher nulos | Estimator |
| `Bucketizer` | numérica | categórica | Bins com limites fixos | Transformer |
| `QuantileDiscretizer` | numérica | categórica | Bins por quantis | Estimator |
| `PCA` | Vector | Vector menor | Redução de dimensão | Estimator |

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

## 5.3 One-hot encoding

O caminho completo de uma coluna de texto até o vetor de features:

```
"Fiber optic"  ──StringIndexer──→  0.0  ──OneHotEncoder──→  [1, 0]
"DSL"          ──StringIndexer──→  1.0  ──OneHotEncoder──→  [0, 1]
"No"           ──StringIndexer──→  2.0  ──OneHotEncoder──→  [0, 0]
                                                             ↑
                                     dropLast=True: a última categoria
                                     vira o vetor de zeros (categoria base)
```

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

df = spark.table("workspace.default.churn_clientes")

indexer = StringIndexer(
    inputCol="internet_service",
    outputCol="internet_idx",
    handleInvalid="keep",     # "error" (default) | "skip" | "keep"
)

encoder = OneHotEncoder(
    inputCol="internet_idx",
    outputCol="internet_ohe",
    dropLast=True,            # default True — evita multicolinearidade
)

modelo = Pipeline(stages=[indexer, encoder]).fit(df)
resultado = modelo.transform(df)

display(resultado.select("internet_service", "internet_idx", "internet_ohe").distinct())
```

### `handleInvalid` — cai na prova

| Valor | Comportamento com categoria não vista no `fit()` |
|---|---|
| `"error"` (default) | Lança exceção |
| `"skip"` | **Remove a linha** do resultado |
| `"keep"` | Cria um índice extra para "desconhecido" e **mantém a linha** |

> Em produção, `"keep"` costuma ser a escolha certa: uma categoria nova aparecer é normal, e derrubar o job ou perder linhas silenciosamente é pior.

### `dropLast` — por que o padrão é `True`

Com N categorias, `dropLast=True` gera N−1 colunas. A categoria descartada é representada pelo vetor de zeros. Isso evita a **armadilha da variável dummy** (multicolinearidade perfeita), que quebra a interpretação de coeficientes em modelos lineares. Para modelos de árvore não faz diferença prática.

> **Quando NÃO usar one-hot encoding** — modelos de árvore, alta cardinalidade e variáveis ordinais. Está detalhado em [3.7](03_eda_preparacao.md#37-quando-one-hot-encoding-é-má-ideia), e é objetivo oficial da Seção 2.

### `StringIndexer` com várias colunas de uma vez

```python
multi_indexer = StringIndexer(
    inputCols=["contract_type", "internet_service", "payment_method"],
    outputCols=["contract_idx", "internet_idx", "payment_idx"],
    handleInvalid="keep",
)

df_indexed = multi_indexer.fit(df).transform(df)
display(df_indexed.select("contract_type", "contract_idx",
                          "internet_service", "internet_idx").limit(5))
```

> Por padrão o `StringIndexer` ordena as categorias por **frequência decrescente** — a categoria mais comum recebe o índice 0. Isso é controlado por `stringOrderType`.

---

## 5.4 Scalers, bins e PCA

### StandardScaler vs MinMaxScaler

```python
from pyspark.ml.feature import StandardScaler, MinMaxScaler

# StandardScaler: (x - média) / desvio → média 0, desvio 1
scaler_std = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True,    # centraliza em zero (subtrai a média)
    withStd=True,     # divide pelo desvio padrão
)

# MinMaxScaler: escala linear para [0, 1]
scaler_mm = MinMaxScaler(
    inputCol="features_raw", outputCol="features_minmax",
    min=0.0, max=1.0,
)
```

| | `StandardScaler` | `MinMaxScaler` |
|---|---|---|
| Resultado | Média 0, desvio 1 | Valores em [0, 1] |
| Range | Ilimitado | Limitado |
| Sensível a outliers | Menos | **Mais** — um único extremo comprime todo o resto |
| Use quando | Distribuição aproximadamente normal; modelos lineares, SVM | Precisa de range fixo; redes neurais |

> **Quem precisa de scaling:** modelos baseados em distância ou gradiente — regressão logística/linear, SVM, KNN, redes neurais, KMeans, PCA. **Quem não precisa:** modelos de árvore (Decision Tree, Random Forest, GBT) são invariantes a transformações monotônicas.

### Bucketizer vs QuantileDiscretizer

```python
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import functions as F

# Bucketizer — VOCÊ define os limites. É um TRANSFORMER (sem fit)
bucketizer = Bucketizer(
    splits=[0, 12, 24, 48, float("inf")],   # N splits → N-1 buckets
    inputCol="tenure_months",
    outputCol="tenure_bucket",
    handleInvalid="keep",
)
df_bucket = bucketizer.transform(df)        # direto, sem fit()

display(
    df_bucket.groupBy("tenure_bucket").agg(
        F.count("*").alias("clientes"),
        F.round(F.mean("churn") * 100, 1).alias("taxa_churn_%"),
    ).orderBy("tenure_bucket")
)
```

```python
# QuantileDiscretizer — o SPARK calcula os limites. É um ESTIMATOR (precisa de fit)
discretizer = QuantileDiscretizer(
    numBuckets=4,                # quartis
    inputCol="monthly_charges",
    outputCol="charges_quartile",
)
df_disc = discretizer.fit(df).transform(df)   # precisa de fit()

display(df_disc.groupBy("charges_quartile")
               .agg(F.count("*").alias("clientes"))
               .orderBy("charges_quartile"))
```

> **A conta que cai na prova:** `Bucketizer` com N splits cria **N−1 buckets**. `splits=[0, 18, 65, inf]` → 3 buckets: `[0,18)`, `[18,65)`, `[65,∞)`.

### PCA

```python
from pyspark.ml.feature import PCA

pca = PCA(k=2, inputCol="features_raw", outputCol="features_pca")
pca_model = pca.fit(train_feat)     # train_feat vem da seção 5.5

print("Variância explicada por componente:")
for i, v in enumerate(pca_model.explainedVariance):
    print(f"  PC{i+1}: {v:.2%}")
print(f"Total: {sum(pca_model.explainedVariance):.2%}")
```

> PCA exige uma coluna **Vector** de entrada, e é **sensível à escala** — padronize antes, senão a feature de maior magnitude domina os componentes.

---

## 5.5 Pipeline completo

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import functions as F
import random

df = spark.table("workspace.default.churn_clientes")

# Injetar alguns nulos para o Imputer ter o que fazer
random.seed(0)
ids_nulos = [f"C{i:04d}" for i in random.sample(range(1, 1001), 50)]
df = df.withColumn(
    "monthly_charges",
    F.when(F.col("customer_id").isin(ids_nulos), None).otherwise(F.col("monthly_charges")),
)

colunas_num = ["tenure_months", "monthly_charges", "total_charges"]
colunas_cat = ["contract_type", "internet_service", "tech_support", "payment_method"]

# PASSO 1 — preencher nulos
imputer = Imputer(
    inputCols=colunas_num,
    outputCols=[c + "_imp" for c in colunas_num],
    strategy="median",
)

# PASSO 2 — texto → índice (SEMPRE antes do OneHotEncoder)
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in colunas_cat
]

# PASSO 3 — índice → vetor binário
encoders = [
    OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe", dropLast=True)
    for c in colunas_cat
]

# PASSO 4 — juntar tudo em UMA coluna Vector (obrigatório no Spark ML)
assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in colunas_num] + [c + "_ohe" for c in colunas_cat],
    outputCol="features_raw",
    handleInvalid="skip",
)

# PASSO 5 — padronizar
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True,
)

pipeline = Pipeline(stages=[imputer] + indexers + encoders + [assembler, scaler])

# CRÍTICO: split ANTES do fit, para evitar data leakage
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

pipeline_model = pipeline.fit(train_df)      # aprende SÓ com o treino
train_feat = pipeline_model.transform(train_df)
test_feat  = pipeline_model.transform(test_df)

display(train_feat.select("customer_id", "features", "churn").limit(5))
print(f"Tamanho do vetor de features: {len(train_feat.select('features').first()[0])}")
```

### A ordem importa

```python
# ERRADO: OneHotEncoder antes do StringIndexer
# OHE exige índices numéricos como entrada — receber texto quebra
Pipeline(stages=[
    OneHotEncoder(inputCol="contract_type", outputCol="contract_ohe"),   # ← erro
    StringIndexer(inputCol="contract_type", outputCol="contract_idx"),
])

# CORRETO
Pipeline(stages=[
    StringIndexer(inputCol="contract_type", outputCol="contract_idx"),   # texto → número
    OneHotEncoder(inputCol="contract_idx",  outputCol="contract_ohe"),   # número → vetor
    VectorAssembler(inputCols=["contract_ohe", "tenure_months"], outputCol="features"),
])
```

### Fit no treino, não no dataset completo

```python
# ERRADO: fit no dataset inteiro → o Imputer e o Scaler "veem" o teste
pipeline_model_errado = pipeline.fit(df)              # ← data leakage

# CORRETO
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
pipeline_model = pipeline.fit(train_df)               # aprende só do treino
train_result = pipeline_model.transform(train_df)
test_result  = pipeline_model.transform(test_df)      # aplica o aprendido do treino
```

> **O Pipeline é a defesa contra data leakage.** Com todos os estágios dentro dele, um único `fit(train_df)` garante que nenhuma estatística do teste vaza para o treino. É por isso que o Pipeline não é só organização de código — é correção metodológica.

---

## 5.6 Pipeline com o modelo dentro

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df = spark.table("workspace.default.churn_clientes")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

imputer   = Imputer(inputCols=["tenure_months", "monthly_charges", "total_charges"],
                    outputCols=["tenure_imp", "monthly_imp", "total_imp"])
indexer   = StringIndexer(inputCol="contract_type", outputCol="contract_idx", handleInvalid="keep")
encoder   = OneHotEncoder(inputCol="contract_idx", outputCol="contract_ohe")
assembler = VectorAssembler(inputCols=["tenure_imp", "monthly_imp", "total_imp", "contract_ohe"],
                            outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features")
lr        = LogisticRegression(featuresCol="features", labelCol="churn")

pipeline = Pipeline(stages=[imputer, indexer, encoder, assembler, scaler, lr])

model = pipeline.fit(train_df)        # treina feature engineering + modelo de uma vez
preds = model.transform(test_df)      # transforma o DF bruto direto em predições

auc = BinaryClassificationEvaluator(labelCol="churn").evaluate(preds)
print(f"AUC-ROC do pipeline completo: {auc:.4f}")
```

> Colocar o modelo dentro do Pipeline permite que o `CrossValidator` (Módulo 6) otimize também os hiperparâmetros dos estágios anteriores, e garante que a feature engineering respeite os folds da validação cruzada.

---

## Exercício prático

```python
# 1. Monte o pipeline da seção 5.5 e treine
# 2. Descubra quantas colunas o vetor final tem — e explique o número
#    (dica: 3 numéricas + soma de (categorias-1) de cada categórica, por causa do dropLast)
# 3. Troque dropLast=True por dropLast=False e veja o tamanho mudar
# 4. Troque StandardScaler por MinMaxScaler e compare os valores do vetor
# 5. Rode o pipeline com handleInvalid="error" no StringIndexer e explique o que acontece
```

---

## Pontos-chave para a prova

- **Estimator tem `fit()` e aprende dos dados; Transformer só tem `transform()`**
- `VectorAssembler` e `Bucketizer` = **Transformers** | `OneHotEncoder`, `StringIndexer`, `Imputer`, `StandardScaler`, `QuantileDiscretizer`, `PCA` = **Estimators**
- Todo modelo treinado (`...Model`) é um **Transformer**
- `Pipeline` é Estimator; `Pipeline.fit()` retorna um **`PipelineModel`** (Transformer)
- Ordem obrigatória: **`StringIndexer` → `OneHotEncoder` → `VectorAssembler`**
- Spark ML exige uma coluna **Vector** (por convenção chamada `features`)
- `handleInvalid`: `"error"` (default) | `"skip"` (remove linha) | `"keep"` (cria índice extra)
- `dropLast=True` (default) evita multicolinearidade em modelos lineares
- `Bucketizer`: N splits → **N−1 buckets**; limites definidos por você
- `QuantileDiscretizer`: `numBuckets=N`; limites calculados pelo Spark
- `StandardScaler` → média 0, desvio 1 | `MinMaxScaler` → [0,1], mais sensível a outliers
- Árvores **não precisam** de scaling; modelos lineares, SVM, KNN e redes neurais precisam
- **Sempre `fit` no treino e `transform` no treino e no teste** — o Pipeline garante isso

---

→ Próximo: [06_treinamento_modelos.md](06_treinamento_modelos.md)
