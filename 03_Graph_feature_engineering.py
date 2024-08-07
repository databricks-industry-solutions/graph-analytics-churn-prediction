# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/graph-analytics-churn-prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineering graph features
# MAGIC
# MAGIC Graph features are generated using Spark GraphFrames with vertex and edge dataframes created in the [EDA](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/3631133442664989/command/3631133442977060) step.

# COMMAND ----------

from graphframes import *
from math import comb
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from warnings import filterwarnings
filterwarnings('ignore', 'DataFrame.sql_ctx is an internal property')

# COMMAND ----------

catalog = "hive_metastore"
db_name = "telco"

# COMMAND ----------

# DBTITLE 1,Read vertex_df
vertex_df = spark.table(f"{catalog}.{db_name}.telco_vertex_df")
display(vertex_df)

# COMMAND ----------

# DBTITLE 1,Read edge_df
edge_df = spark.table(f"{catalog}.{db_name}.telco_edge_df")
display(edge_df)

# COMMAND ----------

# DBTITLE 1,Creating a graph using GraphFrames
g = GraphFrame(vertex_df, edge_df)

# COMMAND ----------

# DBTITLE 1,Degree
# Calculating the number of edges that are connected to each vertex.

degree_df = g.degrees
graph_features_df = vertex_df.alias('customer').join(degree_df, degree_df.id == vertex_df.id, 'left')\
                             .select('customer.id', 'degree')\
                             .withColumnRenamed('id','customer_id')\
                             .fillna(0, "degree")
          
display(graph_features_df.orderBy(F.col("degree").desc()))

# COMMAND ----------

# DBTITLE 1,In-degree
# Calculating the number of edges that are directed towards each vertex.

indegree_df = g.inDegrees
graph_features_df = graph_features_df.alias('features').join(indegree_df, indegree_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'inDegree')\
                 .fillna(0, "inDegree")\
                 .withColumnRenamed("inDegree","in_degree")
display(graph_features_df.orderBy(F.col("inDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Out-degree
# Calculating the number of edges that are originated from each vertex.

outdegree_df = g.outDegrees
graph_features_df = graph_features_df.alias('features').join(outdegree_df, outdegree_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'outDegree')\
                 .fillna(0, "outDegree")\
                 .withColumnRenamed("outDegree","out_degree")
display(graph_features_df.orderBy(F.col("outDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Degree ratio
# Calculating degree ratio

def degreeRatio(x, d):
  if d==0:
    return 0.0
  else:
    return x/d
  
degreeRatioUDF = udf(degreeRatio, FloatType())   

graph_features_df = graph_features_df.withColumn("in_degree_ratio", degreeRatioUDF(F.col("in_degree"), F.col("degree")))
graph_features_df = graph_features_df.withColumn("out_degree_ratio", degreeRatioUDF(F.col("out_degree"), F.col("degree")))
display(graph_features_df)

# COMMAND ----------

# MAGIC %md ## PageRank
# MAGIC
# MAGIC PageRank is a measure of the importance or centrality of a node in a graph, originally developed by Larry Page and Sergey Brin while they were studying at Stanford University. It is used by the Google search engine to rank web pages in its search results.
# MAGIC
# MAGIC The PageRank of a node in a graph is based on the idea that a node's importance is determined by the number and quality of the incoming links it receives from other nodes in the graph. In other words, the more incoming links a node has from other important nodes, the more important it is considered to be.
# MAGIC
# MAGIC The PageRank algorithm assigns a score to each node in the graph based on this idea. The score of a node is calculated iteratively, by considering the scores of all the nodes that link to it, and the scores of all the nodes that those nodes link to, and so on. The algorithm uses a damping factor to prevent the score of a node from becoming too large, and it terminates after a fixed number of iterations or when the scores converge.
# MAGIC
# MAGIC The PageRank score of a node can be used to rank the nodes in the graph by importance or centrality. Nodes with higher PageRank scores are considered to be more important or central to the graph. The PageRank algorithm is widely used in network analysis and information retrieval, and has been extended to many other applications beyond the web.
# MAGIC
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/pagerank.png?raw=true" width="600" />

# COMMAND ----------

# DBTITLE 0,PageRank
# Calculating pagerank

pr_df = g.pageRank(resetProbability=0.15, tol=0.01).vertices.select('id','pagerank')
graph_features_df = graph_features_df.alias('features').join(pr_df, pr_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'pagerank')

display(graph_features_df)

# COMMAND ----------

# MAGIC %md ###Triangle count
# MAGIC
# MAGIC Computes the number of triangles passing through each vertex.

# COMMAND ----------

# DBTITLE 0,Triangle Count
# Calculating triangle count

trian_count = g.triangleCount()

graph_features_df = graph_features_df.alias('features').join(trian_count.select('id','count'), trian_count.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'count')\
                 .withColumnRenamed("count","trian_count")

display(graph_features_df.orderBy(F.col("trian_count").desc()))

# COMMAND ----------

# MAGIC %md #### Clustering coefficient
# MAGIC
# MAGIC The clustering coefficient of a node in a graph is a measure of the degree to which the neighbors of the node are connected to each other. It is defined as the ratio of the number of edges between the neighbors of the node to the maximum number of edges that could exist between them.
# MAGIC
# MAGIC Clustering coefficient of a given node is defined as: 
# MAGIC $$ cc(i) = {\text{Number of complete triangles with coner } i \over \text{Number of all triangular graphs with coner } i} $$
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/clustering_coefficient.png?raw=true" width="800" />

# COMMAND ----------

# DBTITLE 0,Clustering coefficient 
# Calculating clustering coefficient 

def clusterCoefficient(t, e):
  if e==0 or t==0:
    return 0.0
  else:
    return t/comb(e, 2)
  
clusterCoefficientUDF = udf(clusterCoefficient, FloatType())   

graph_features_df = graph_features_df.withColumn("cc", clusterCoefficientUDF(F.col("trian_count"), F.col("degree")))
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df.orderBy(F.col("degree").desc()))

# COMMAND ----------

# MAGIC %md ## Community Detection using Label Propagation
# MAGIC
# MAGIC Run static Label Propagation Algorithm for detecting communities in networks.
# MAGIC
# MAGIC Each node in the network is initially assigned to its own community. At every superstep, nodes send their community affiliation to all neighbors and update their state to the most frequent community affiliation of incoming messages.
# MAGIC
# MAGIC LPA is a standard community detection algorithm for graphs. It is very inexpensive computationally, although (1) convergence is not guaranteed and (2) one can end up with trivial solutions (all nodes are identified into a single community).

# COMMAND ----------

# DBTITLE 0,Community detection
communities = g.labelPropagation(maxIter=25)
display(communities)

# COMMAND ----------

# DBTITLE 1,Calculating community stats
comm_avg = communities.groupBy('label')\
                    .agg(F.avg("monthly_charges").alias("comm_avg_monthly_charges"), \
                         F.avg("total_charges").alias("comm_avg_total_charges"), \
                         F.avg("tenure").alias("comm_avg_tenure"), \
                         F.count("id").alias("comm_size")) 
display(comm_avg)

# COMMAND ----------

# DBTITLE 1,Deviation with average community values
communities = communities.join(comm_avg, on='label', how='left')
communities = communities.withColumn('comm_dev_avg_monthly_charges', F.col('comm_avg_monthly_charges')-F.col('monthly_charges'))
communities = communities.withColumn('comm_dev_avg_total_charges', F.col('comm_avg_total_charges')-F.col('total_charges'))
communities = communities.withColumn('comm_dev_avg_tenure', F.col('comm_avg_tenure')-F.col('tenure'))
display(communities)

# COMMAND ----------

graph_features_df = graph_features_df.alias('features')\
                 .join(communities.alias('comm'),\
                       communities.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'comm.comm_avg_monthly_charges', 'comm.comm_avg_total_charges', 'comm.comm_avg_tenure', 'comm.comm_size',\
                        'comm.comm_dev_avg_monthly_charges', 'comm.comm_dev_avg_total_charges', 'comm.comm_dev_avg_tenure') 
display(graph_features_df)

# COMMAND ----------

# DBTITLE 1,Calculating neighbour averages
edge_df_1 = edge_df.withColumnRenamed('src','id').withColumnRenamed('dst','nbgh')
edge_df_2 = edge_df.withColumnRenamed('dst','id').withColumnRenamed('src','nbgh')
und_edge_df = edge_df_1.union(edge_df_1)
und_edge_df = und_edge_df.alias('edge').join(vertex_df.select('id', 'monthly_charges', 'total_charges', 'tenure').alias('vertex'),\
                              und_edge_df.nbgh==vertex_df.id, how='left')\
                              .select('edge.*', 'vertex.monthly_charges', 'vertex.total_charges', 'vertex.tenure')\
                              .groupBy('id')\
                                  .agg(F.avg("monthly_charges").alias("nghb_avg_monthly_charges"), \
                                       F.avg("total_charges").alias("nghb_avg_total_charges"), \
                                       F.avg("tenure").alias("nghb_avg_tenure")) 
graph_features_df = graph_features_df.alias('features')\
                 .join(und_edge_df.alias('nbgh'),\
                       und_edge_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'nbgh.nghb_avg_monthly_charges', 'nbgh.nghb_avg_total_charges', 'nbgh.nghb_avg_tenure') 
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Report

# COMMAND ----------

display(graph_features_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Write to Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  #drop table if exists
  fs.drop_table(f"{catalog}.{db_name}.telco_churn_graph_features")
except:
  pass

#Note: You might need to delete the FS table using the UI
graph_feature_table = fs.create_table(
  name=f'{catalog}.{db_name}.telco_churn_graph_features',
  primary_keys='customer_id',
  schema=graph_features_df.schema,
  description='These features are derived from the telco customer call network.'
)

fs.write_table(df=graph_features_df, name=f"{catalog}.{db_name}.telco_churn_graph_features", mode='overwrite')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using Databricks AutoML to build our model
# MAGIC
# MAGIC Next step: [Churn preiction model using AutoML]($./04_AutoML_churn_prediction)

# COMMAND ----------


