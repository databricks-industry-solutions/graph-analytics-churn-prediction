# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/graph-analytics-churn-prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC # XGBoost Classifier training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **13.3.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/898470372133770).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "churn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="a9b370dc14dd4567b3b0ac88899f8a22", artifact_path="data", dst_path=input_temp_dir)

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
supported_cols = ["nghb_avg_total_charges", "online_security_no", "online_backup_yes", "phone_service_yes", "out_degree_ratio", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "comm_avg_tenure", "in_degree", "tech_support_no", "tenure", "senior_citizen_1", "multiple_lines_yes", "payment_method_credit_card__automatic_", "online_backup_no", "payment_method_mailed_check", "phone_service_no", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "nghb_avg_monthly_charges", "streaming_movies_no_internet_service", "comm_dev_avg_total_charges", "internet_service_dsl", "payment_method_electronic_check", "degree", "multiple_lines_no_phone_service", "comm_dev_avg_monthly_charges", "comm_avg_total_charges", "tech_support_no_internet_service", "trian_count", "device_protection_no_internet_service", "contract_two_year", "device_protection_no", "payment_method_bank_transfer__automatic_", "device_protection_yes", "dependents_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "cc", "comm_size", "contract_month_to_month", "contract_one_year", "out_degree", "streaming_movies_no", "in_degree_ratio", "online_security_yes", "online_security_no_internet_service", "multiple_lines_no", "monthly_charges", "senior_citizen_0", "total_charges", "pagerank", "comm_dev_avg_tenure", "internet_service_fiber_optic", "nghb_avg_tenure", "gender_male", "internet_service_no", "partner_yes", "comm_avg_monthly_charges"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boolean columns
# MAGIC For each column, impute missing values and then convert into ones and zeros.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


bool_imputers = []

bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
    ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
])

bool_transformers = [("boolean", bool_pipeline, ["online_security_no", "online_backup_yes", "phone_service_yes", "streaming_tv_yes", "dependents_no", "online_backup_no_internet_service", "tech_support_no", "multiple_lines_yes", "payment_method_credit_card__automatic_", "senior_citizen_1", "online_backup_no", "phone_service_no", "payment_method_mailed_check", "partner_no", "streaming_tv_no_internet_service", "paperless_billing_yes", "streaming_movies_no_internet_service", "internet_service_dsl", "payment_method_electronic_check", "multiple_lines_no_phone_service", "contract_two_year", "device_protection_no", "tech_support_no_internet_service", "device_protection_no_internet_service", "payment_method_bank_transfer__automatic_", "dependents_yes", "device_protection_yes", "tech_support_yes", "streaming_movies_yes", "streaming_tv_no", "gender_female", "paperless_billing_no", "contract_month_to_month", "contract_one_year", "streaming_movies_no", "online_security_yes", "online_security_no_internet_service", "senior_citizen_0", "multiple_lines_no", "internet_service_fiber_optic", "gender_male", "internet_service_no", "partner_yes"])]

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
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
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

from databricks.automl_runtime.sklearn import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["comm_size"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = bool_transformers + numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_0000` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/898470372133770)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from databricks.automl_runtime.sklearn import TransformedTargetClassifier

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

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

def objective(params):
  with mlflow.start_run(experiment_id="898470372133770") as mlflow_run:
    xgbc_classifier = TransformedTargetClassifier(
        classifier=XGBClassifier(**params),
        transformer=LabelEncoder()  # XGBClassifier requires the target values to be integers between 0 and n_class-1
    )

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", xgbc_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True)

    model.fit(X_train, y_train, classifier__early_stopping_rounds=5, classifier__verbose=False, classifier__eval_set=[(X_val_processed,y_val_processed)])

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": 1 }
    )
    xgbc_training_metrics = training_eval_result.metrics
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_" , "pos_label": 1 }
    )
    xgbc_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": 1 }
    )
    xgbc_test_metrics = test_eval_result.metrics

    loss = -xgbc_val_metrics["val_roc_auc"]

    # Truncate metric key names so they can be displayed together
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": xgbc_val_metrics,
      "test_metrics": xgbc_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

space = {
  "colsample_bytree": 0.29914794927133104,
  "learning_rate": 0.06689877153734992,
  "max_depth": 4,
  "min_child_weight": 19,
  "n_estimators": 256,
  "n_jobs": 100,
  "subsample": 0.3799214775881078,
  "verbosity": 0,
  "random_state": 495337292,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=1,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patch pandas version in logged model
# MAGIC
# MAGIC Ensures that model serving uses the same version of pandas that was used to train the model.

# COMMAND ----------

import mlflow
import os
import shutil
import tempfile
import yaml

run_id = mlflow_run.info.run_id

# Set up a local dir for downloading the artifacts.
tmp_dir = str(tempfile.TemporaryDirectory())
os.makedirs(tmp_dir)

client = mlflow.tracking.MlflowClient()

# Fix conda.yaml
conda_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir)
with open(conda_file_path) as f:
  conda_libs = yaml.load(f, Loader=yaml.FullLoader)
pandas_lib_exists = any([lib.startswith("pandas==") for lib in conda_libs["dependencies"][-1]["pip"]])
if not pandas_lib_exists:
  print("Adding pandas dependency to conda.yaml")
  conda_libs["dependencies"][-1]["pip"].append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/conda.yaml", "w") as f:
    f.write(yaml.dump(conda_libs))
  client.log_artifact(run_id=run_id, local_path=conda_file_path, artifact_path="model")

# Fix requirements.txt
venv_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir)
with open(venv_file_path) as f:
  venv_libs = f.readlines()
venv_libs = [lib.strip() for lib in venv_libs]
pandas_lib_exists = any([lib.startswith("pandas==") for lib in venv_libs])
if not pandas_lib_exists:
  print("Adding pandas dependency to requirements.txt")
  venv_libs.append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/requirements.txt", "w") as f:
    f.write("\n".join(venv_libs))
  client.log_artifact(run_id=run_id, local_path=venv_file_path, artifact_path="model")

shutil.rmtree(tmp_dir)

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
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=495337292)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=495337292)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
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
# MAGIC model_uri=f"models:/{model_name}/{model_version}"
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
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

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/898470372133770/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

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

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))
