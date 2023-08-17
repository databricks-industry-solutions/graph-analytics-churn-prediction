# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/graph-analytics-churn-prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineering customer features
# MAGIC
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

from graphframes import *
from math import comb
import re

# COMMAND ----------

catalog = "hive_metastore"
db_name = "telco"

# COMMAND ----------

# DBTITLE 1,Read in Silver Delta table using Spark
# Read customer data into Spark
customer_df = spark.table(f"{catalog}.{db_name}.telco_churn_customers_silver")
display(customer_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Pandas on spark
# MAGIC
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `Pandas on spark` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `pandas_api`.*

# COMMAND ----------

# DBTITLE 1,Define customer featurization function
from databricks.feature_store import feature_table
import pyspark.pandas as ps

def compute_customer_features(data):
  
  # Convert to a dataframe compatible with the pandas API
  data = data.pandas_api()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents', 'senior_citizen',
                                 'phone_service', 'multiple_lines', 'internet_service',
                                 'online_security', 'online_backup', 'device_protection',
                                 'tech_support', 'streaming_tv', 'streaming_movies',
                                 'contract', 'paperless_billing', 'payment_method'], dtype = 'int64')
  
  # Convert label to int and rename column
  data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churn': 'int32'})
  
  # Clean up column names
  data.columns = [re.sub(r'[\(\)]', ' ', name).lower() for name in data.columns]
  data.columns = [re.sub(r'[ -]', '_', name).lower() for name in data.columns]

  
  # Drop missing values
  data = data.dropna()
  
  return data

customer_df = customer_df.drop('mobile_number')
customer_features_df = compute_customer_features(customer_df)

# COMMAND ----------

display(customer_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Report

# COMMAND ----------

display(customer_features_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Write to Feature Store
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  #drop table if exists
  fs.drop_table(f"{catalog}.{db_name}.telco_churn_customer_features")
except:
  pass
#Note: You might need to delete the FS table using the UI
customer_feature_table = fs.create_table(
  name=f"{catalog}.{db_name}.telco_churn_customer_features",
  primary_keys='customer_id',
  schema=customer_features_df.spark.schema(),
  description='These features are derived from the telco_churn_customers_silver table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=customer_features_df.to_spark(), name=f"{catalog}.{db_name}.telco_churn_customer_features", mode='overwrite')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Graph Feature Engineering
# MAGIC Our next job is to prepare a set of features from the customer call graph that we'll be able to use in customer churn prediction and other data science projects.
# MAGIC
# MAGIC
# MAGIC Next: [Graph feature engineering]($./03_Graph_feature_engineering)
