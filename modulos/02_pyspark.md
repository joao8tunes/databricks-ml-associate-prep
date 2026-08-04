# Módulo 2 — PySpark para ML

> **Notebook sugerido:** `02_pyspark_para_ml`
>
> **O que você aprende:** DataFrames Spark, transformações essenciais, EDA, lazy evaluation.
>
> Se você conhece pandas, vai reconhecer quase tudo. A diferença é que os dados ficam distribuídos entre os nós do cluster — o Spark cuida disso automaticamente.

---

## 2.1 Criar o dataset do projeto

Este dataset é usado em todos os módulos seguintes, salvo como tabela gerenciada no Unity Catalog (`workspace.default.churn_clientes`). Execute uma vez e reutilize.

```python
import random
from pyspark.sql.types import *
from pyspark.sql import functions as F

random.seed(42)
n = 1000
dados = []

for i in range(n):
    tenure          = random.randint(1, 72)
    contract        = random.choice(["Month-to-month", "One year", "Two year"])
    monthly         = round(random.uniform(20, 120), 2)
    internet        = random.choice(["DSL", "Fiber optic", "No"])
    tech_support    = random.choice(["Yes", "No"])
    payment         = random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    # Lógica realista: churn mais provável em contratos mensais e contas altas
    prob = 0.4 if contract == "Month-to-month" else 0.1
    prob += 0.1 if monthly > 80 else 0
    prob -= 0.1 if tenure > 24 else 0
    churn = 1 if random.random() < max(0.05, min(0.8, prob)) else 0

    dados.append((
        f"C{i+1:04d}", tenure, monthly,
        round(monthly * tenure, 2),
        contract, internet, tech_support, payment, churn
    ))

schema = StructType([
    StructField("customer_id",     StringType(),  False),
    StructField("tenure_months",   IntegerType(), True),
    StructField("monthly_charges", DoubleType(),  True),
    StructField("total_charges",   DoubleType(),  True),
    StructField("contract_type",   StringType(),  True),
    StructField("internet_service",StringType(),  True),
    StructField("tech_support",    StringType(),  True),
    StructField("payment_method",  StringType(),  True),
    StructField("churn",           IntegerType(), False),
])

df = spark.createDataFrame(dados, schema)

TABELA = "workspace.default.churn_clientes"
df.write.format("delta").mode("overwrite").saveAsTable(TABELA)

display(df)
print(f"\nTotal: {df.count()} clientes")
print(f"Taxa de churn: {df.filter(F.col('churn') == 1).count() / df.count():.1%}")
```

---

## 2.2 PySpark vs Pandas — tabela de equivalências

| Operação | Pandas | PySpark |
|---|---|---|
| Ler CSV | `pd.read_csv("f.csv")` | `spark.read.csv("f.csv", header=True, inferSchema=True)` |
| Ver schema | `df.dtypes` | `df.printSchema()` |
| Ver dados | `df.head(5)` | `df.show(5)` ou `display(df)` |
| Filtrar | `df[df["col"] > 5]` | `df.filter(F.col("col") > 5)` |
| Selecionar colunas | `df[["a", "b"]]` | `df.select("a", "b")` |
| Nova coluna | `df["c"] = df["a"] + df["b"]` | `df.withColumn("c", F.col("a") + F.col("b"))` |
| Group By | `df.groupby("col").mean()` | `df.groupBy("col").mean()` |
| Contar linhas | `len(df)` | `df.count()` |
| Converter | — | `df.toPandas()` — **cuidado: traz tudo para memória** |
| Renomear | `df.rename(columns={"a": "b"})` | `df.withColumnRenamed("a", "b")` |
| Remover coluna | `df.drop("col", axis=1)` | `df.drop("col")` |
| Ordenar | `df.sort_values("col", ascending=False)` | `df.orderBy(F.col("col").desc())` |

---

## 2.3 Exploração de Dados (EDA)

```python
df = spark.table("workspace.default.churn_clientes")

# 1. Estrutura
df.printSchema()

# 2. Estatísticas descritivas
display(df.describe())

# 3. Contagem de nulos por coluna
from pyspark.sql.functions import col, count, when, isnan

nulos = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in df.columns
])
display(nulos)

# 4. Distribuição da variável alvo
display(
    df.groupBy("churn")
      .agg(count("*").alias("qtd"))
      .withColumn("percentual", F.round(F.col("qtd") / df.count() * 100, 1))
)

# 5. Churn por tipo de contrato (análise de negócio)
display(
    df.groupBy("contract_type").agg(
        count("*").alias("total"),
        F.sum("churn").alias("churned"),
        F.round(F.mean("churn") * 100, 1).alias("taxa_churn_%")
    ).orderBy("taxa_churn_%", ascending=False)
)

# 6. Correlação entre variáveis numéricas e churn
for col_name in ["tenure_months", "monthly_charges", "total_charges"]:
    corr = df.stat.corr(col_name, "churn")
    print(f"Correlação {col_name} ↔ churn: {corr:.3f}")
```

---

## 2.4 Transformações Essenciais

```python
from pyspark.sql import functions as F

df = spark.table("workspace.default.churn_clientes")

# --- Criar novas colunas ---
df = df.withColumn(
    "avg_monthly_spend",
    F.round(F.col("total_charges") / F.col("tenure_months"), 2)
)

df = df.withColumn(
    "faixa_tenure",
    F.when(F.col("tenure_months") <= 12, "0-12 meses")
     .when(F.col("tenure_months") <= 36, "13-36 meses")
     .otherwise("37+ meses")
)

df = df.withColumn(
    "is_premium",
    F.when(F.col("monthly_charges") >= 80, 1).otherwise(0)
)

# --- Filtros ---
# Clientes de alto risco (contrato mensal + pouco tempo + conta alta)
alto_risco = df.filter(
    (F.col("contract_type") == "Month-to-month") &
    (F.col("tenure_months") < 12) &
    (F.col("monthly_charges") > 60)
)
print(f"Clientes alto risco: {alto_risco.count()}")

# --- Agregações ---
display(
    df.groupBy("contract_type", "internet_service").agg(
        count("*").alias("total"),
        F.round(F.mean("monthly_charges"), 2).alias("mensalidade_media"),
        F.round(F.mean("churn") * 100, 1).alias("taxa_churn_%")
    ).orderBy("taxa_churn_%", ascending=False)
)

# --- Join com tabela externa ---
dados_suporte = [("C0001", 5, "VIP"), ("C0002", 0, "Standard"), ("C0003", 12, "VIP")]
df_suporte = spark.createDataFrame(dados_suporte, ["customer_id", "chamados", "tier"])
df_enrich  = df.join(df_suporte, on="customer_id", how="left")
df_enrich  = df_enrich.fillna({"chamados": 0, "tier": "Standard"})

# --- Operações com strings ---
df = df.withColumn("payment_short", F.split(F.col("payment_method"), " ")[0])

# --- Operações com datas ---
from pyspark.sql.functions import current_date, datediff, to_date
df = df.withColumn("data_referencia", current_date())

# --- Remover duplicatas ---
df_sem_dup = df.dropDuplicates(["customer_id"])
print(f"Após dedup: {df_sem_dup.count()}")

# --- Preencher nulos ---
df = df.fillna({
    "monthly_charges": df.agg(F.mean("monthly_charges")).collect()[0][0],
    "contract_type": "Month-to-month"
})

# --- Ordenar e pegar top N ---
display(
    df.orderBy(F.col("monthly_charges").desc())
      .select("customer_id", "monthly_charges", "churn")
      .limit(10)
)
```

---

## 2.5 Funções de Coluna — Referência Rápida

```python
from pyspark.sql import functions as F

# Matemática
F.abs(col)         # valor absoluto
F.round(col, 2)    # arredondar
F.sqrt(col)        # raiz quadrada
F.log(col)         # logaritmo natural
F.pow(col, 2)      # potência

# Strings
F.upper(col)                   # maiúsculas
F.lower(col)                   # minúsculas
F.trim(col)                    # remover espaços
F.length(col)                  # tamanho da string
F.split(col, ",")              # dividir string → array
F.concat(col1, F.lit("-"), col2)  # concatenar
F.regexp_replace(col, r"\s+", "_")  # regex

# Datas
F.year(col)                    # extrair ano
F.month(col)                   # extrair mês
F.dayofweek(col)               # dia da semana (1=Dom)
F.datediff(col1, col2)         # diferença em dias
F.date_add(col, 30)            # adicionar dias
F.current_timestamp()          # timestamp atual

# Condicionais
F.when(cond, val).otherwise(outro)
F.coalesce(col1, col2)         # primeiro não-nulo
F.isnull(col)                  # True se nulo
F.isnan(col)                   # True se NaN

# Agregações
F.count("*")
F.sum(col)
F.mean(col) / F.avg(col)
F.stddev(col)
F.min(col) / F.max(col)
F.collect_list(col)            # agrupa valores em lista
F.collect_set(col)             # agrupa valores únicos em lista
F.countDistinct(col)           # conta valores únicos
```

---

## 2.6 Lazy Evaluation — Conceito Crítico

```python
# IMPORTANTE: transformações Spark NÃO executam imediatamente.
# Elas constroem um plano de execução (DAG) que só roda quando uma ACTION é chamada.

# Estas 3 linhas NÃO fazem nada ainda (transformações = lazy):
df2 = df.filter(F.col("churn") == 1)
df2 = df2.withColumn("flag_alto_valor", F.col("monthly_charges") > 100)
df2 = df2.select("customer_id", "monthly_charges", "flag_alto_valor")

# Esta linha EXECUTA tudo de uma vez (action = eager):
resultado = df2.count()  # ← aqui o Spark processa todas as transformações acima

# Outras actions comuns:
df.show(5)            # imprime no console
df.collect()          # traz TODAS as linhas para o driver (⚠️ cuidado em dados grandes)
df.toPandas()         # converte para pandas (⚠️ traz tudo para memória)
df.write.save(...)    # escreve em disco
df.first()            # retorna a primeira linha
df.take(10)           # retorna as 10 primeiras linhas como lista

# Cache: quando você usa o mesmo DataFrame várias vezes
df_filtrado = df.filter(F.col("churn") == 1).cache()  # ← armazena em memória
print(df_filtrado.count())   # processa e armazena
print(df_filtrado.count())   # usa o cache (muito mais rápido!)
df_filtrado.unpersist()      # libera o cache
```

> **Regra prática:** Só use `.collect()` ou `.toPandas()` em dados pequenos — resultados de agregações ou amostras. Em dados grandes, mantenha tudo como Spark DataFrame.

---

## 2.7 Split treino/teste no Spark

```python
# Spark usa randomSplit (não train_test_split do sklearn)
df = spark.table("workspace.default.churn_clientes")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Treino: {train_df.count()} | Teste: {test_df.count()}")

# Verificar balanceamento
print(f"Churn no treino: {train_df.filter(F.col('churn')==1).count()/train_df.count():.1%}")
print(f"Churn no teste:  {test_df.filter(F.col('churn')==1).count()/test_df.count():.1%}")
```

---

## Pontos-chave para a prova

- `display(df)` → visualização rica (específica do Databricks)
- Transformações são **lazy** — nada executa até uma action ser chamada
- Actions: `count()`, `show()`, `collect()`, `first()`, `write`
- `collect()` traz tudo para o driver → evitar em dados grandes
- `randomSplit([0.8, 0.2], seed=42)` → retorna lista de DataFrames
- `fillna({"col": valor})` → preenche nulos em colunas específicas
- `filter()` e `where()` são equivalentes no Spark

---

→ Próximo: [03_mlflow.md](03_mlflow.md)
