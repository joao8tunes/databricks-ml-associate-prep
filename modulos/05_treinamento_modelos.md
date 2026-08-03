# Módulo 5 — Treinamento de Modelos

> **Notebook sugerido:** `05_treinamento_modelos`
>
> Spark MLlib tem modelos para classificação, regressão e clustering. A API é idêntica ao sklearn: `fit()` treina, `transform()` prediz.

---

## 5.1 Modelos disponíveis no Spark MLlib

```python
# --- Classificação ---
from pyspark.ml.classification import (
    LogisticRegression,              # rápido, bom baseline
    RandomForestClassifier,          # robusto, não precisa de normalização
    GBTClassifier,                   # Gradient Boosted Trees (muito usado)
    DecisionTreeClassifier,          # interpretável
    LinearSVC,                       # SVM linear
    NaiveBayes,                      # bom para texto
    MultilayerPerceptronClassifier,  # rede neural simples
)

# --- Regressão ---
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
    DecisionTreeRegressor,
    GeneralizedLinearRegression,  # GLM (família Poisson, Gamma, etc.)
)

# --- Clustering ---
from pyspark.ml.clustering import (
    KMeans,
    BisectingKMeans,    # hierárquico divisivo (mais rápido que KMeans)
    GaussianMixture,
    LDA,                # Latent Dirichlet Allocation (para texto)
)

# --- Recomendação ---
from pyspark.ml.recommendation import ALS  # Alternating Least Squares
```

---

## 5.2 Requisitos dos modelos Spark ML

```
Todo modelo Spark ML espera:
├── featuresCol → coluna do tipo Vector (criada pelo VectorAssembler)
├── labelCol    → coluna numérica (IntegerType ou DoubleType)
└── predictionCol → onde o modelo escreve a predição (default: "prediction")

Colunas que o modelo ADICIONA ao DataFrame:
├── rawPrediction → scores brutos (log-odds para LR, contagens para RF)
├── probability   → probabilidade por classe (ex: [0.3, 0.7])
└── prediction    → classe predita (0 ou 1)
```

---

## 5.3 Fluxo completo de treinamento + avaliação

```python
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
import mlflow

df = spark.read.format("delta").load("dbfs:/FileStore/churn_project/data/clientes")

# --- Pipeline de feature engineering (do Módulo 4) ---
colunas_num = ["tenure_months", "monthly_charges", "total_charges"]
colunas_cat = ["contract_type", "internet_service", "tech_support", "payment_method"]

stages = [
    Imputer(inputCols=colunas_num, outputCols=[c+"_imp" for c in colunas_num]),
    *[StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in colunas_cat],
    *[OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe") for c in colunas_cat],
    VectorAssembler(
        inputCols=[c+"_imp" for c in colunas_num] + [c+"_ohe" for c in colunas_cat],
        outputCol="features_raw"
    ),
    StandardScaler(inputCol="features_raw", outputCol="features"),
]

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
feat_pipeline = Pipeline(stages=stages)
feat_model    = feat_pipeline.fit(train_df)
train_feat    = feat_model.transform(train_df)
test_feat     = feat_model.transform(test_df)

# --- Treinar RandomForest ---
mlflow.set_experiment("/churn-prediction-sparkml")

with mlflow.start_run(run_name="RF_Spark"):
    mlflow.log_param("framework", "Spark MLlib")
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 5)

    rf = RandomForestClassifier(
        featuresCol="features",   # nome padrão esperado pelos modelos
        labelCol="churn",         # label numérica obrigatória
        numTrees=100,
        maxDepth=5,
        seed=42
    )

    rf_model    = rf.fit(train_feat)
    predictions = rf_model.transform(test_feat)
    # predictions tem: features, churn, rawPrediction, probability, prediction

    # --- Avaliar: BinaryClassificationEvaluator ---
    bin_eval = BinaryClassificationEvaluator(
        labelCol="churn",
        rawPredictionCol="rawPrediction",  # default
        metricName="areaUnderROC"          # "areaUnderROC" | "areaUnderPR"
    )
    auc = bin_eval.evaluate(predictions)

    # --- Avaliar: MulticlassClassificationEvaluator ---
    mc_eval = MulticlassClassificationEvaluator(
        labelCol="churn",
        predictionCol="prediction"
    )
    acc = mc_eval.setMetricName("accuracy").evaluate(predictions)
    f1  = mc_eval.setMetricName("f1").evaluate(predictions)
    pr  = mc_eval.setMetricName("weightedPrecision").evaluate(predictions)
    rec = mc_eval.setMetricName("weightedRecall").evaluate(predictions)

    mlflow.log_metrics({"auc_roc": auc, "accuracy": acc, "f1": f1})

    # Logar modelo Spark ML
    mlflow.spark.log_model(rf_model, artifact_path="spark_rf_model")

    print(f"AUC-ROC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
```

---

## 5.4 Ver as predições detalhadas

```python
# Extrair probabilidade da classe positiva (churn=1)
display(
    predictions
    .withColumn("prob_churn", F.round(F.col("probability")[1], 4))
    .select("customer_id", "churn", "prediction", "prob_churn")
    .orderBy(F.col("prob_churn").desc())
    .limit(15)
)

# Feature Importance do RandomForest
feature_names = (
    colunas_num +
    [f"{c}_{cat}" for c in colunas_cat for cat in ["0", "1", "2"]]  # aproximação
)
print("\nFeature Importances:")
for name, imp in sorted(zip(feature_names, rf_model.featureImportances),
                         key=lambda x: -x[1])[:5]:
    print(f"  {name:35s}: {imp:.4f}")
```

---

## 5.5 GBTClassifier — Gradient Boosted Trees

```python
with mlflow.start_run(run_name="GBT_Spark"):
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="churn",
        maxIter=50,        # número de árvores
        maxDepth=4,
        stepSize=0.1,      # learning rate
        seed=42
    )

    gbt_model = gbt.fit(train_feat)
    preds_gbt = gbt_model.transform(test_feat)

    auc_gbt = bin_eval.evaluate(preds_gbt)
    acc_gbt = mc_eval.setMetricName("accuracy").evaluate(preds_gbt)

    mlflow.log_params({"maxIter": 50, "maxDepth": 4, "stepSize": 0.1})
    mlflow.log_metrics({"auc_roc": auc_gbt, "accuracy": acc_gbt})
    mlflow.spark.log_model(gbt_model, "spark_gbt_model")

    print(f"GBT - AUC-ROC: {auc_gbt:.4f} | Accuracy: {acc_gbt:.4f}")

# ATENÇÃO: GBTClassifier NÃO produz coluna "probability"
# Ele produz apenas "rawPrediction" e "prediction"
# Use rawPrediction com BinaryClassificationEvaluator
```

---

## 5.6 Regressão com Spark ML

```python
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Prever mensalidade (monthly_charges) como exemplo de regressão
# Atenção: não faz sentido de negócio real, mas demonstra a API

gbt_reg = GBTRegressor(
    featuresCol="features",
    labelCol="monthly_charges",  # target numérico
    maxIter=50,
    maxDepth=4,
    seed=42
)

gbt_reg_model = gbt_reg.fit(train_feat)
preds_reg     = gbt_reg_model.transform(test_feat)

reg_eval = RegressionEvaluator(
    labelCol="monthly_charges",
    predictionCol="prediction"
)

print(f"RMSE: {reg_eval.setMetricName('rmse').evaluate(preds_reg):.4f}")
print(f"R²:   {reg_eval.setMetricName('r2').evaluate(preds_reg):.4f}")
print(f"MAE:  {reg_eval.setMetricName('mae').evaluate(preds_reg):.4f}")
```

---

## 5.7 Cross-Validation

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit

# Modelo base para tuning
rf_cv = RandomForestClassifier(featuresCol="features", labelCol="churn", seed=42)

# Definir grade de parâmetros
param_grid = (ParamGridBuilder()
    .addGrid(rf_cv.numTrees, [50, 100, 200])
    .addGrid(rf_cv.maxDepth, [3, 5, 8])
    .build()
)
# 3 × 3 = 9 combinações

evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")

# --- CrossValidator: k-fold (mais robusto, mais lento) ---
cv = CrossValidator(
    estimator=rf_cv,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,   # 3-fold cross-validation
    seed=42
)

print("Treinando CrossValidator (pode demorar)...")
cv_model    = cv.fit(train_feat)
best_rf_cv  = cv_model.bestModel

print(f"Melhor AUC (CV médio): {max(cv_model.avgMetrics):.4f}")
print(f"numTrees: {best_rf_cv.getNumTrees}")
print(f"maxDepth: {best_rf_cv.getOrDefault('maxDepth')}")

# Avaliar no teste
preds_cv = cv_model.transform(test_feat)
print(f"AUC no teste: {evaluator.evaluate(preds_cv):.4f}")
```

```python
# --- TrainValidationSplit: split único (mais rápido) ---
tvs = TrainValidationSplit(
    estimator=rf_cv,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8,   # 80% treino, 20% validação interna
    seed=42
)

tvs_model = tvs.fit(train_feat)
print(f"Melhor AUC (TVS): {max(tvs_model.validationMetrics):.4f}")

# Diferença CrossValidator vs TrainValidationSplit:
# CrossValidator    → k-fold, mais robusto, k × n_combinações modelos treinados
# TrainValidationSplit → 1 split, mais rápido, 1 × n_combinações modelos
```

---

## 5.8 Pipeline completo: feature eng + modelo

```python
# O ideal é incluir o modelo DENTRO do pipeline
# Assim, cross-validation otimiza também os stages anteriores

from pyspark.ml.classification import LogisticRegression

pipeline_full = Pipeline(stages=[
    Imputer(inputCols=colunas_num, outputCols=[c+"_imp" for c in colunas_num]),
    *[StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in colunas_cat],
    *[OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe") for c in colunas_cat],
    VectorAssembler(
        inputCols=[c+"_imp" for c in colunas_num] + [c+"_ohe" for c in colunas_cat],
        outputCol="features_raw"
    ),
    StandardScaler(inputCol="features_raw", outputCol="features"),
    LogisticRegression(featuresCol="features", labelCol="churn"),  # modelo DENTRO do pipeline
])

model_full = pipeline_full.fit(train_df)   # treina tudo junto no DF bruto
preds_full = model_full.transform(test_df) # transforma o DF bruto diretamente

auc_full = evaluator.evaluate(preds_full)
print(f"Pipeline completo AUC-ROC: {auc_full:.4f}")
```

---

## Avaliadores — resumo completo para a prova

```python
# CLASSIFICAÇÃO BINÁRIA
from pyspark.ml.evaluation import BinaryClassificationEvaluator
eval_bin = BinaryClassificationEvaluator(labelCol="label")
eval_bin.setMetricName("areaUnderROC")   # default
eval_bin.setMetricName("areaUnderPR")    # área sob curva precisão-recall

# CLASSIFICAÇÃO MULTICLASSE (funciona para binário também)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
eval_mc = MulticlassClassificationEvaluator(labelCol="label")
eval_mc.setMetricName("accuracy")
eval_mc.setMetricName("f1")
eval_mc.setMetricName("weightedPrecision")
eval_mc.setMetricName("weightedRecall")
eval_mc.setMetricName("truePositiveRateByLabel")  # recall por classe

# REGRESSÃO
from pyspark.ml.evaluation import RegressionEvaluator
eval_reg = RegressionEvaluator(labelCol="label")
eval_reg.setMetricName("rmse")   # root mean squared error (default)
eval_reg.setMetricName("mse")    # mean squared error
eval_reg.setMetricName("mae")    # mean absolute error
eval_reg.setMetricName("r2")     # R² (coeficiente de determinação)
eval_reg.setMetricName("var")    # variância explicada

# CLUSTERING
from pyspark.ml.evaluation import ClusteringEvaluator
eval_clust = ClusteringEvaluator()
eval_clust.setMetricName("silhouette")  # silhouette score (default)
```

---

## Pontos-chave para a prova

- Spark ML exige coluna `features` do tipo **Vector** (criada pelo VectorAssembler)
- `labelCol` deve ser numérico — use `IntegerType` ou `DoubleType`
- `predictions.select("probability")[1]` → probabilidade da classe positiva
- `GBTClassifier` **não** produz coluna `probability`, só `rawPrediction`
- `CrossValidator` → k-fold (mais robusto)
- `TrainValidationSplit` → split único (mais rápido)
- `BinaryClassificationEvaluator` → `areaUnderROC` ou `areaUnderPR`
- `MulticlassClassificationEvaluator` → `accuracy`, `f1`, `weightedPrecision`, `weightedRecall`
- `RegressionEvaluator` → `rmse`, `mse`, `mae`, `r2`
- Incluir o modelo no Pipeline → CV otimiza os estágios anteriores também

---

→ Próximo: [06_hyperopt.md](06_hyperopt.md)
