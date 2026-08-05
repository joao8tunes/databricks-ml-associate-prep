# Módulo 0 — Configuração do Ambiente

> **Objetivo:** Ter o ambiente Databricks funcionando antes de iniciar os estudos.

---

## 0.1 Criar conta gratuita no Databricks (Free Edition)

O Databricks oferece uma conta gratuita chamada **Free Edition** (antiga Community Edition) — sem cartão de crédito e sem prazo de expiração.

**Passo a passo:**

1. Acesse: **https://www.databricks.com/learn/free-edition**
2. Clique em **"Sign up for free"**
3. Preencha nome, e-mail e senha
4. Verifique seu e-mail e clique no link de confirmação
5. Faça login em **https://login.databricks.com**

### Limitações da Free Edition

| Recurso | Free Edition | Conta Paga |
|---|---|---|
| Compute | Somente **Serverless** — sem criar/configurar cluster | Clusters clássicos (All-Purpose, Job) configuráveis |
| AutoML (classificação/regressão) | ❌ Não disponível (exige cluster clássico) | ✅ Disponível |
| Model Serving (REST) | ✅ Disponível, com limites (nº de endpoints, sem GPU, sem provisioned throughput) | ✅ Completo |
| Feature Store | ✅ Disponível | ✅ Completo |
| MLflow | ✅ Completo | ✅ Completo |
| Spark ML | ✅ Completo (via Serverless) | ✅ Completo |

> O tópico de AutoML é cobrado na prova mesmo sem conseguir rodar no Free Edition — estude o código.

---

## 0.2 Compute: por que não existe "Create Cluster" aqui

Diferente de contas pagas, o Free Edition **não permite criar nem configurar clusters**. Todo o compute é **Serverless**: o Databricks aloca e gerencia as máquinas automaticamente por trás dos panos, sem tela de "Create Cluster", sem escolher tamanho de máquina, sem esperar o cluster subir.

Isso significa:
- Notebooks conectam a **Serverless** automaticamente — não existe passo de criar cluster.
- O SQL Editor/Dashboards usam o **"Serverless Starter Warehouse"**, o único SQL Warehouse disponível na conta gratuita.
- Você não escolhe versão de Databricks Runtime — o ambiente serverless já vem com Python, Spark, MLflow, sklearn, XGBoost etc. pré-instalados e atualizados pelo Databricks.

> **Para a prova:** o conceito de cluster (All-Purpose vs Job Cluster, Databricks Runtime, Runtime ML) continua sendo cobrado — veja o Módulo 1. Aqui você só não vai *praticar* a criação manual, porque o Free Edition abstrai isso.

---

## 0.3 Criar o primeiro notebook

1. Menu esquerdo → **"Workspace"** → **"Users"** → sua pasta
2. Botão direito → **"Create"** → **"Notebook"**
3. Configure:
   - **Name:** `00_verificacao_ambiente`
   - **Language:** Python
4. Clique em **"Create"**
5. No canto superior esquerdo do notebook, confirme que o seletor de compute mostra **"Serverless"** (conecta sozinho ao rodar a primeira célula — não precisa fazer nada)

**Cole e execute este código de verificação:**

```python
import pyspark
import mlflow
import sklearn

print(f"PySpark:    {pyspark.__version__}")
print(f"MLflow:     {mlflow.__version__}")
print(f"Sklearn:    {sklearn.__version__}")
print(f"\nSpark ativo: {spark}")
```

Se os 4 itens aparecerem sem erro, o ambiente está pronto.

---

## 0.4 Atalhos essenciais do notebook

| Atalho | Ação |
|---|---|
| `Shift + Enter` | Executar célula e avançar |
| `Ctrl + Enter` | Executar célula e ficar |
| `Esc + A` | Nova célula acima |
| `Esc + B` | Nova célula abaixo |
| `%sql` (início da célula) | Escrever SQL |
| `%md` (início da célula) | Escrever Markdown |
| `display(df)` | Visualização rica de DataFrame |

---

## Próximo módulo

→ [01_plataforma.md](01_plataforma.md) — Clusters, Unity Catalog & Volumes, Delta Lake, dbutils
