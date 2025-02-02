# Tax Filling Retention

A FastAPI application to recommend properties based on the property and user data. 

# Pre-reqs(poetry installation): Optional

### Install the [poetry](https://python-poetry.org/docs/#installation)

NOTE : if you don't want to use poetry you can manually install packages listed in pyproject.toml

# Installation

1. clone and cd into the repo
2. Run

    ```shell
    poetry install
    ```

    to install packages
3. Run

    ```shell
    poetry shell
    ```

    to to activate the environment

4. Run the application using:

   ```shell
    uvicorn taxfilingretention.app:app --port 9070 --env-file .env
    ```
    OR

    Use docker compose to run the system

   ```shell
    docker compose --env-file .env -f docker-compose.yaml build
    ```
    and to run the application container

   ```shell
    docker compose --env-file .env -f docker-compose.yaml build
    ```


🔧 Next Steps
- Hyperparameter tuning for XGBoost
- Create a separate Service for training (that can be used for retraining based on the new data)
    -   Use MLFlow to evaluate and select the best model
- Deploying API to cloud (AWS/GCP/Azure) - Both training and inference piepline
- Logging & monitoring for inference requests
    - Prometheus and Grafana for monitoring
    - Locust for SLA(s)
    - A simple logger is already included that can be used based on the requirement
- Adding feature importance visualization

🛠️ Tech Stack
- Python (pandas, scikit-learn, XGBoost)
- FastAPI (for API serving)
- Joblib (for model persistence)
- Uvicorn (for ASGI server)