# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **nuwan-ML** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/4126749559790436/s?orderByKey=metrics.%60val_roc_auc_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/4126749559778792) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC
# MAGIC Runtime Version: _11.2.x-gpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "churn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("d3f74daef3a940d38e50fe900cf7f029", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["nghb_avg_total_charges", "online_security_no", "online_backup_yes", "phone_service_yes", "out_degree_ratio", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "comm_avg_tenure", "in_degree", "tenure", "tech_support_no", "senior_citizen_1", "multiple_lines_yes", "payment_method_credit_card__automatic_", "online_backup_no", "payment_method_mailed_check", "phone_service_no", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "nghb_avg_monthly_charges", "streaming_movies_no_internet_service", "comm_dev_avg_total_charges", "internet_service_dsl", "payment_method_electronic_check", "degree", "multiple_lines_no_phone_service", "comm_dev_avg_monthly_charges", "comm_avg_total_charges", "trian_count", "tech_support_no_internet_service", "device_protection_no_internet_service", "contract_two_year", "device_protection_no", "payment_method_bank_transfer__automatic_", "dependents_yes", "device_protection_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "cc", "comm_size", "contract_month_to_month", "out_degree", "contract_one_year", "streaming_movies_no", "in_degree_ratio", "online_security_yes", "online_security_no_internet_service", "monthly_charges", "multiple_lines_no", "senior_citizen_0", "total_charges", "pagerank", "comm_dev_avg_tenure", "internet_service_fiber_optic", "nghb_avg_tenure", "gender_male", "internet_service_no", "partner_yes", "comm_avg_monthly_charges"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["cc", "comm_avg_monthly_charges", "comm_avg_tenure", "comm_avg_total_charges", "comm_dev_avg_monthly_charges", "comm_dev_avg_tenure", "comm_dev_avg_total_charges", "comm_size", "contract_month_to_month", "contract_one_year", "contract_two_year", "degree", "dependents_no", "dependents_yes", "device_protection_no", "device_protection_no_internet_service", "device_protection_yes", "gender_female", "gender_male", "in_degree", "in_degree_ratio", "internet_service_dsl", "internet_service_fiber_optic", "internet_service_no", "monthly_charges", "multiple_lines_no", "multiple_lines_no_phone_service", "multiple_lines_yes", "nghb_avg_monthly_charges", "nghb_avg_tenure", "nghb_avg_total_charges", "online_backup_no", "online_backup_no_internet_service", "online_backup_yes", "online_security_no", "online_security_no_internet_service", "online_security_yes", "out_degree", "out_degree_ratio", "pagerank", "paperless_billing_no", "paperless_billing_yes", "partner_no", "partner_yes", "payment_method_bank_transfer__automatic_", "payment_method_credit_card__automatic_", "payment_method_electronic_check", "payment_method_mailed_check", "phone_service_no", "phone_service_yes", "senior_citizen_0", "senior_citizen_1", "streaming_movies_no", "streaming_movies_no_internet_service", "streaming_movies_yes", "streaming_tv_no", "streaming_tv_no_internet_service", "streaming_tv_yes", "tech_support_no", "tech_support_no_internet_service", "tech_support_yes", "tenure", "total_charges", "trian_count"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["nghb_avg_total_charges", "online_security_no", "online_backup_yes", "phone_service_yes", "out_degree_ratio", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "in_degree", "tech_support_no", "comm_avg_tenure", "tenure", "payment_method_credit_card__automatic_", "multiple_lines_yes", "senior_citizen_1", "phone_service_no", "payment_method_mailed_check", "online_backup_no", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "nghb_avg_monthly_charges", "streaming_movies_no_internet_service", "comm_dev_avg_total_charges", "internet_service_dsl", "payment_method_electronic_check", "degree", "multiple_lines_no_phone_service", "comm_dev_avg_monthly_charges", "comm_avg_total_charges", "trian_count", "tech_support_no_internet_service", "device_protection_no_internet_service", "contract_two_year", "device_protection_no", "payment_method_bank_transfer__automatic_", "device_protection_yes", "dependents_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "cc", "comm_size", "contract_month_to_month", "out_degree", "contract_one_year", "streaming_movies_no", "in_degree_ratio", "online_security_yes", "online_security_no_internet_service", "monthly_charges", "multiple_lines_no", "senior_citizen_0", "total_charges", "pagerank", "comm_dev_avg_tenure", "internet_service_fiber_optic", "nghb_avg_tenure", "gender_male", "internet_service_no", "partner_yes", "comm_avg_monthly_charges"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["comm_size", "contract_month_to_month", "contract_one_year", "contract_two_year", "dependents_no", "dependents_yes", "device_protection_no", "device_protection_no_internet_service", "device_protection_yes", "gender_female", "gender_male", "internet_service_dsl", "internet_service_fiber_optic", "internet_service_no", "multiple_lines_no", "multiple_lines_no_phone_service", "multiple_lines_yes", "online_backup_no", "online_backup_no_internet_service", "online_backup_yes", "online_security_no", "online_security_no_internet_service", "online_security_yes", "paperless_billing_no", "paperless_billing_yes", "partner_no", "partner_yes", "payment_method_bank_transfer__automatic_", "payment_method_credit_card__automatic_", "payment_method_electronic_check", "payment_method_mailed_check", "phone_service_no", "phone_service_yes", "senior_citizen_0", "senior_citizen_1", "streaming_movies_no", "streaming_movies_no_internet_service", "streaming_movies_yes", "streaming_tv_no", "streaming_tv_no_internet_service", "streaming_tv_yes", "tech_support_no", "tech_support_no_internet_service", "tech_support_yes"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_9832` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_9832 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_9832 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_9832 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_9832 == "test"]

# Separate target column from features and drop _automl_split_col_9832
X_train = split_train_df.drop([target_col, "_automl_split_col_9832"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_9832"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_9832"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4126749559790436/s?orderByKey=metrics.%60val_roc_auc_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from databricks.automl_runtime.sklearn import TransformedTargetClassifier

set_config(display="diagram")

xgbc_classifier = TransformedTargetClassifier(
    classifier=XGBClassifier(
        colsample_bytree=0.34227087299460235,
        learning_rate=0.03211290116495768,
        max_depth=5,
        min_child_weight=19,
        n_estimators=277,
        n_jobs=100,
        subsample=0.41771684410577314,
        verbosity=0,
        random_state=383591162,
    ),
    transformer=LabelEncoder()  # XGBClassifier requires the target values to be integers between 0 and n_class-1
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("classifier", xgbc_classifier),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)
label_encoder_val = LabelEncoder()
label_encoder_val.fit(y_train)
y_val_processed = label_encoder_val.transform(y_val)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(experiment_id="4126749559790436", run_name="xgboost") as mlflow_run:
    model.fit(X_train, y_train, classifier__early_stopping_rounds=5, classifier__verbose=False, classifier__eval_set=[(X_val_processed,y_val_processed)])
    
    # Log metrics for the training set
    xgbc_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_", pos_label=1)

    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_", pos_label=1)

    # Log metrics for the test set
    xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_", pos_label=1)

    # Display the logged metrics
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    display(pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=383591162)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=383591162)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Paste the entire output (%md ...) to an empty cell, and click the link to see the MLflow run page
print(f"%md [Link to model run page](#mlflow/experiments/4126749559790436/runs/{ mlflow_run.info.run_id }/artifactPath/model)")

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve.png")
display(Image(filename=eval_pr_curve_path))
