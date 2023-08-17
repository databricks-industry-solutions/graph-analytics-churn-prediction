# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/graph-analytics-churn-prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC ![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)
# MAGIC
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
# MAGIC [![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC
# MAGIC ## Graph analytics for telco customer churn prediction
# MAGIC <List of the business use case the solution accelerator address>
# MAGIC
# MAGIC Graph analytics is a field of data analysis that focuses on extracting insights from data represented as graphs. A graph is a mathematical representation of a network of interconnected objects, where the objects are represented as nodes, and the connections between them are represented as edges.
# MAGIC
# MAGIC Features extracted from telco customer network can provide valuable insights into the relationships and patterns of behavior among customers. Customer relationship data can be represented as a graph, where nodes represent customers and edges represent phone calls between customers.
# MAGIC
# MAGIC By analyzing the call network graph features, machine learning models can identify patterns and predict which customers are most likely to churn. For example, machine learning models can analyze the network structure to identify customers who are more central or connected in the network, indicating that they may have a greater influence on other customers' behavior. Additionally, machine learning models can analyze the patterns of calls between customers, such as the frequency and duration of calls.
# MAGIC
# MAGIC By combining these features with other customer data, such as demographics and usage patterns, machine learning models can build more accurate models for predicting customer churn. This can enable telecom companies to take proactive steps to retain customers and improve the customer experience, ultimately leading to increased customer loyalty and profitability.
# MAGIC <img src="https://github.com/nuwan-db/Graph_Analytics_Telco_Churn_Prediction/blob/dev/_resources/images/overview.png?raw=true" width="1000" />
# MAGIC ___
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  
# MAGIC
# MAGIC ## Getting started
# MAGIC
# MAGIC Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 
# MAGIC
# MAGIC <img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">
# MAGIC
# MAGIC To start using a solution accelerator in Databricks simply follow these steps: 
# MAGIC
# MAGIC 1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
# MAGIC 2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
# MAGIC 3. Execute the multi-step-job to see how the pipeline runs. 
# MAGIC 4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.
# MAGIC
# MAGIC The cost associated with running the accelerator is the user's responsibility.
# MAGIC
# MAGIC
# MAGIC ## Project support 
# MAGIC
# MAGIC Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
# MAGIC
