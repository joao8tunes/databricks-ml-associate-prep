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
| Cluster | Single Node (1 máquina) | Multi-Node (cluster real) |
| AutoML | ❌ Não disponível | ✅ Disponível |
| Model Serving (REST) | ❌ Não disponível | ✅ Disponível |
| Feature Store | ✅ Limitado | ✅ Completo |
| MLflow | ✅ Completo | ✅ Completo |
| Spark ML | ✅ Completo | ✅ Completo |

> Os tópicos de AutoML e Model Serving são cobrados na prova — estude o código mesmo sem conseguir rodar localmente.

---

## 0.2 Criar o primeiro cluster

1. No menu esquerdo, clique em **"Compute"**
2. Clique em **"Create Cluster"**
3. Configure:
   - **Cluster name:** `ml-study-cluster`
   - **Cluster mode:** Single Node
   - **Databricks Runtime:** versão mais recente com **"ML"** no nome — ex: `14.x ML`
4. Clique em **"Create Cluster"**
5. Aguarde o símbolo verde (cluster pronto — pode levar 3–5 min)

> **Runtime ML vs Runtime padrão:**
> - Runtime padrão: Spark + Python
> - Runtime ML: tudo do padrão + MLflow, sklearn, XGBoost, LightGBM, PyTorch, TensorFlow
>
> **Sempre use Runtime ML** para os exercícios deste guia.

---

## 0.3 Criar o primeiro notebook

1. Menu esquerdo → **"Workspace"** → **"Users"** → sua pasta
2. Botão direito → **"Create"** → **"Notebook"**
3. Configure:
   - **Name:** `00_verificacao_ambiente`
   - **Language:** Python
   - **Cluster:** selecione o cluster criado acima
4. Clique em **"Create"**

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

→ [01_plataforma.md](01_plataforma.md) — Clusters, DBFS, Delta Lake, dbutils
