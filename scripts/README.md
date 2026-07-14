# CLI & Automation Scripts

Helper scripts to execute model training runs and predictions locally and on the cluster:

- `train_ru.py`: Execution entrypoint on the Ramanujan Universe HPC Cluster (Linux) to train LSTM, Transformer, and XGBoost models. Output weights, scaling parameters, and target features configs are stored under `saved_models/`.
- `predict.py`: Evaluates predictions on local instances.
