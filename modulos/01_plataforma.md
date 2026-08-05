# Módulo 1 — Plataforma Databricks

> **Notebook sugerido:** `01_plataforma_databricks`
>
> **O que você aprende:** Clusters, Unity Catalog & Volumes, Delta Lake, dbutils — a base de tudo no Databricks.

---

## 1.1 Clusters

```
Cluster = conjunto de máquinas que executam seu código Spark.

Arquitetura:
┌─────────────────────────────────────┐
│           DRIVER NODE               │  ← seu código roda aqui
│  (orquestra, coleta resultados)     │
└──────────────┬──────────────────────┘
               │ distribui tarefas
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Worker 1   Worker 2   Worker 3   ← processam dados em paralelo
```

> **Analogia:** se **Cluster** é o hardware (as máquinas — quantas, de que tamanho), **Databricks Runtime** é o sistema operacional + programas pré-instalados nelas (Spark, e no caso do Runtime ML, também MLflow, sklearn, XGBoost...). Você escolhe o hardware (cluster) e o software que vem instalado nele (runtime) separadamente.

### Tipos de cluster — o que cai na prova

| Tipo | Uso | Custo | Quando usar |
|---|---|---|---|
| **All-Purpose** | Notebooks interativos | Maior (fica ligado) | Desenvolvimento |
| **Job Cluster** | Jobs agendados | Menor (cria e destrói automaticamente) | Produção |

> **Na prova:** Job Cluster é sempre a resposta certa para produção — é destruído ao fim do job, pagando só pelo tempo de execução.

> ⚠️ **No Free Edition você não cria nem configura clusters** — só existe compute **Serverless**, que abstrai tudo isso (o Databricks decide o hardware por trás dos panos). A tabela acima é teoria de prova; para praticar, seus notebooks já conectam em Serverless automaticamente (veja o Módulo 0).

### Databricks Runtime

```
Runtime 14.x          → Python + Spark padrão
Runtime 14.x ML       → + MLflow, sklearn, XGBoost, PyTorch, TensorFlow
Runtime 14.x ML GPU   → + suporte a GPU (deep learning com GPU)
```

No Serverless não existe seletor de versão de Runtime — o Databricks mantém o ambiente (equivalente ao Runtime ML) atualizado automaticamente.

---

## 1.2 DBFS — Databricks File System (teoria para a prova)

DBFS é o sistema de arquivos distribuído do Databricks, montado sobre o object storage da cloud. Persiste entre sessões (diferente do sistema de arquivos local do driver) e é acessível em todos os clusters do workspace. Caminhos começam com `dbfs:/`, e `dbfs:/FileStore/` era a área tradicional de upload.

```
dbutils.fs.ls("dbfs:/")             → listar arquivos
dbutils.fs.mkdirs("dbfs:/caminho")  → criar pasta
dbutils.fs.cp / mv / rm             → copiar, mover, remover
dbfs:/FileStore/                    → área tradicional de upload de arquivos
```

> ⚠️ **No Databricks Free Edition, o "public DBFS root" vem desabilitado por padrão.** Rodar `dbutils.fs.ls("dbfs:/")` ainda funciona, mas só mostra 3 pastas especiais montadas (`Volumes/`, `Workspace/`, `databricks-datasets/`) — não é um espaço livre para gravar. Qualquer tentativa de **escrever** fora desses mounts (ex: `dbutils.fs.mkdirs("dbfs:/caminho")` ou qualquer coisa em `dbfs:/FileStore/`) retorna o erro `DBFS_DISABLED: Public DBFS root is disabled`. Isso não é um bug: é a direção atual do Databricks (inclusive em contas pagas) de migrar armazenamento de arquivos para o **Unity Catalog**. Guarde o conceito de DBFS para a prova — ainda é cobrado — mas os exercícios práticos deste guia usam Unity Catalog a partir daqui.

---

## 1.2b Unity Catalog — armazenamento na prática

O Unity Catalog organiza dados em três níveis: **catalog → schema → tabela/volume** (equivalente a database → schema → tabela no mundo relacional). Toda conta Free Edition já vem com o catalog `workspace` e o schema `default` prontos para uso, então não é preciso criar nada extra para começar.

### DBFS vs. Unity Catalog — o que é cada coisa

| | DBFS (legado) | Unity Catalog |
|---|---|---|
| **O que é** | Um sistema de arquivos único por workspace | Um catálogo de dados com governança: `catalog.schema.tabela` e `/Volumes/catalog/schema/volume/` |
| **Como você referencia algo** | Path (`dbfs:/pasta/arquivo.csv`) | Nome lógico (`workspace.default.tabela`) ou path de Volume |
| **Permissões** | Por workspace inteiro, pouco granular | Granular — por catalog, schema, tabela ou volume |
| **Funciona entre workspaces?** | Não | Sim (um metastore pode ser compartilhado) |
| **Neste guia** | Só teoria (cai na prova) | O que você usa na prática |
| **Regra prática** | Se você pensou em `dbfs:/algo`, é aqui que o Free Edition bloqueia | Tabela gerenciada para Delta, Volume para arquivo cru |

| Recurso Unity Catalog | Para quê | Substitui |
|---|---|---|
| **Tabela gerenciada** | Salvar DataFrames como Delta — forma recomendada, sem lidar com paths | `spark.write.save(path)` em `dbfs:/` |
| **Volume** | Arquivos crus: CSV, imagens, checkpoints de streaming | `dbfs:/FileStore/` |

O exemplo abaixo é autocontido — cria um DataFrame pequeno só para você testar o padrão agora. **O dado real do projeto (tabela `churn_clientes`) só é criado no Módulo 2** — aqui é só para praticar a sintaxe.

```python
# Criar um DataFrame de exemplo (não é o dado do projeto — é só para testar)
df_exemplo = spark.createDataFrame([("C001", 0), ("C002", 1)], ["customer_id", "churn"])

# Tabela gerenciada — não precisa de path, só catalog.schema.tabela
df_exemplo.write.format("delta").mode("overwrite").saveAsTable("workspace.default.exemplo_uc")

# Ler de volta
display(spark.table("workspace.default.exemplo_uc"))
# → deve mostrar as 2 linhas que você acabou de escrever
```

Para arquivos crus (o que antes ia em `dbfs:/FileStore/`), crie um volume uma única vez e use `dbutils.fs` normalmente — a API não muda, só o caminho:

```sql
-- Rodar uma vez (célula %sql ou spark.sql)
CREATE VOLUME IF NOT EXISTS workspace.default.churn_project;
```

O volume começa **vazio**. Os comandos abaixo criam um arquivo de teste primeiro e depois manipulam esse mesmo arquivo, para você ver o efeito de cada operação:

```python
CAMINHO = "/Volumes/workspace/default/churn_project"

# 0. Criar um arquivo de teste (senão não existe nada para listar/copiar/mover)
dbutils.fs.put(f"{CAMINHO}/origem.csv", "customer_id,churn\nC001,0\nC002,1\n", overwrite=True)

# 1. Listar arquivos → deve aparecer "origem.csv"
display(dbutils.fs.ls(CAMINHO))

# 2. Criar pasta → cria "data/", ainda vazia
dbutils.fs.mkdirs(f"{CAMINHO}/data/")

# 3. Copiar → agora existem "origem.csv" E "destino.csv" (os dois, lado a lado)
dbutils.fs.cp(f"{CAMINHO}/origem.csv", f"{CAMINHO}/destino.csv")

# 4. Mover → "destino.csv" desaparece da raiz e passa a existir em "data/arquivo.csv"
dbutils.fs.mv(f"{CAMINHO}/destino.csv", f"{CAMINHO}/data/arquivo.csv")

# 5. Deletar um arquivo → remove só "origem.csv"
dbutils.fs.rm(f"{CAMINHO}/origem.csv")

# 6. Deletar uma pasta inteira (recursivo) → remove "data/" e o "arquivo.csv" dentro dela
dbutils.fs.rm(f"{CAMINHO}/data/", recurse=True)

# Conferir que o volume voltou a ficar vazio
display(dbutils.fs.ls(CAMINHO))
```

---

## 1.3 Delta Lake

Delta Lake é o formato padrão de armazenamento no Databricks. Pense nele como um Parquet com superpoderes.

> O exemplo abaixo cria um dataset pequeno e fictício **só para praticar Delta Lake** — não é o dataset do projeto. O dataset real (`churn_clientes`, ~1000 clientes) é criado no Módulo 2 e é o que os módulos seguintes usam.

```
Delta Lake resolve problemas do CSV/Parquet puro:
├── ACID transactions → escrita atômica, sem corrupção em falhas
├── Time Travel       → voltar para qualquer versão anterior
├── Schema enforcement → rejeita dados que não batem com o schema
└── Otimizações       → Z-ordering, compaction, vacuum automáticos
```

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F

# Criar dados de exemplo
schema = StructType([
    StructField("customer_id",   StringType(),  False),
    StructField("tenure_months", IntegerType(), True),
    StructField("monthly_charges", DoubleType(), True),
    StructField("churn",         IntegerType(), False),
])

dados = [
    ("C001", 12, 65.50, 1),
    ("C002", 24, 89.99, 0),
    ("C003",  3, 45.00, 1),
    ("C004", 60, 110.0, 0),
]

df = spark.createDataFrame(dados, schema)

# Salvar como tabela Delta gerenciada (Unity Catalog)
TABELA = "workspace.default.demo_delta"  # nome de demonstração — não confundir com "churn_clientes" do Módulo 2
df.write.format("delta").mode("overwrite").saveAsTable(TABELA)

# Ler Delta
df_delta = spark.table(TABELA)
display(df_delta)
```

```python
# --- Adicionar dados (nova versão) ---
novos = spark.createDataFrame([("C005", 8, 55.0, 1), ("C006", 36, 95.0, 0)], schema)
novos.write.format("delta").mode("append").saveAsTable(TABELA)

# --- Time Travel: ver versões anteriores ---
# Listar histórico
display(spark.sql(f"DESCRIBE HISTORY {TABELA}"))

# Ler versão específica
df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABELA)
print(f"Versão 0: {df_v0.count()} registros")

# Ler versão por timestamp
# df_ts = spark.read.format("delta").option("timestampAsOf", "2024-01-01").table(TABELA)
```

```python
# --- Consultar com SQL ---
# saveAsTable já registrou a tabela — dá pra usar SQL direto
display(spark.sql(f"SELECT * FROM {TABELA} WHERE churn = 1"))

# Ou misturar Python e SQL no mesmo notebook
# %sql
# SELECT tenure_months, COUNT(*) as total, AVG(churn) as taxa_churn
# FROM workspace.default.demo_delta
# GROUP BY tenure_months
```

---

## 1.4 dbutils — utilitário especial do Databricks

```python
# dbutils.fs      → operações de arquivo (já vimos acima)
# dbutils.secrets → acessar credenciais com segurança
# dbutils.widgets → parâmetros interativos
# dbutils.notebook → chamar outros notebooks

# --- Widgets: parâmetros interativos ---
dbutils.widgets.text("modelo", "RandomForest", "Tipo de Modelo")
dbutils.widgets.dropdown("n_trees", "100", ["50", "100", "200"], "Nº de Árvores")
dbutils.widgets.slider("test_size", 0.2, 0.1, 0.4, 0.05, "Tamanho do Teste")

# Ler valor dos widgets
modelo = dbutils.widgets.get("modelo")
n_trees = int(dbutils.widgets.get("n_trees"))
print(f"Modelo: {modelo} | Árvores: {n_trees}")

# --- Secrets: acessar tokens/senhas de forma segura ---
# (sem expor no código)
# token = dbutils.secrets.get(scope="meu-scope", key="api-token")

# --- Chamar outro notebook e passar o resultado ---
# resultado = dbutils.notebook.run("../outro_notebook", timeout_seconds=600,
#                                   arguments={"param1": "valor"})

# Limpar widgets ao final
dbutils.widgets.removeAll()
```

---

## Exercício Prático

Execute em um notebook novo no Databricks:

```python
from pyspark.sql.types import *
from pyspark.sql import functions as F

# 1. Criar e salvar dados como Delta
schema = StructType([
    StructField("produto", StringType()),
    StructField("preco", DoubleType()),
    StructField("estoque", IntegerType()),
])

produtos = [
    ("Notebook", 3500.0, 10),
    ("Mouse", 80.0, 150),
    ("Teclado", 200.0, 75),
    ("Monitor", 1200.0, 30),
]

df = spark.createDataFrame(produtos, schema)
TABELA_EX = "workspace.default.exercicio_delta"
df.write.format("delta").mode("overwrite").saveAsTable(TABELA_EX)

# 2. Adicionar novos produtos (nova versão)
novos = spark.createDataFrame([("Headset", 350.0, 45)], schema)
novos.write.format("delta").mode("append").saveAsTable(TABELA_EX)

# 3. Ver histórico de versões
display(spark.sql(f"DESCRIBE HISTORY {TABELA_EX}"))

# 4. Comparar versões
df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABELA_EX)
df_v1 = spark.table(TABELA_EX)
print(f"Versão 0: {df_v0.count()} produtos | Versão 1: {df_v1.count()} produtos")
```

---

## Pontos-chave para a prova

- Cluster = hardware (máquinas); Runtime = software instalado nelas (Spark + libs)
- Job Cluster = criado e destruído automaticamente → mais barato para produção
- Runtime ML = inclui MLflow, sklearn, PyTorch, TensorFlow pré-instalados
- Serverless = Databricks gerencia o hardware por você — sem criar/configurar cluster (padrão no Free Edition)
- DBFS = sistema de arquivos distribuído, persiste entre sessões (cai na prova; root público desabilitado no Free Edition)
- Unity Catalog = catalog → schema → tabela/volume; substitui o DBFS na prática (tabelas gerenciadas + Volumes)
- Delta Lake = ACID + time travel + schema enforcement
- `option("versionAsOf", N)` = lê versão N do histórico Delta
- `DESCRIBE HISTORY` = lista todas as versões de uma tabela Delta

---

→ Próximo: [02_pyspark.md](02_pyspark.md)
