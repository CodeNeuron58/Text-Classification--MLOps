import dagshub
dagshub.init(repo_owner='CodeNeuron58', repo_name='Text-Classification--MLOps', mlflow=True)


import mlflow

mlflow.set_tracking_uri('https://dagshub.com/CodeNeuron58/Text-Classification--MLOps.mlflow')
mlflow.set_experiment('SetUp Check')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)