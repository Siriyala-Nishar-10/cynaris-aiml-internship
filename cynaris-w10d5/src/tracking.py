import os

def log_run(category: str, priority: str, status: str) -> None:
    if os.getenv('ENABLE_MLFLOW', 'false').lower() != 'true':
        return
    import mlflow
    mlflow.set_experiment('cynaris-w10d5-customer-support')
    with mlflow.start_run():
        mlflow.log_params({'category': category, 'priority': priority})
        mlflow.log_metric('human_review_required', int(status in {'approved', 'escalated'}))
        mlflow.set_tag('workflow_status', status)
