# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/graph-analytics-churn-prediction.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC
# MAGIC Bootstraping new ML projects can still be long and inefficient. 
# MAGIC
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC
# MAGIC
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`churn_features`)
# MAGIC
# MAGIC Our prediction target is the `churn` column.
# MAGIC
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

from databricks import automl, feature_store

# COMMAND ----------

catalog = "hive_metastore"
db_name = "telco"

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

customer_features = fs.read_table(name=f"{catalog}.{db_name}.telco_churn_customer_features")

# COMMAND ----------

graph_features = fs.read_table(name=f"{catalog}.{db_name}.telco_churn_graph_features")

# COMMAND ----------

features = customer_features.join(graph_features, on='customer_id', how='left')

# COMMAND ----------

summary = automl.classify(
    features,
    target_col="churn",
    exclude_frameworks=["sklearn","lightgbm"],
    exclude_cols=["customer_id"],
    primary_metric="roc_auc",
    timeout_minutes=5
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using the generated notebook to build our model
# MAGIC
# MAGIC Next step: [Explore the generated Auto-ML sample notebook]($./05_AutoML_generated_notebook)

# COMMAND ----------


