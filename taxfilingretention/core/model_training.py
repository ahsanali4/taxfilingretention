from typing import List

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


class Classifier:
    def __init__(self) -> None:
        # Define XGBoost classifier with desired hyperparameters
        self.xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=200,  # Number of trees
            learning_rate=0.05,  # Step size shrinkage
            max_depth=6,  # Tree depth
            subsample=0.8,  # Use 80% of the data per tree
            colsample_bytree=0.8,  # Use 80% of features for each tree
            random_state=42,
        )

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """Train the XGBoost classifier.

        Args:
            X_train (np.ndarray): scaled features of the user(s)
            y_train (np.ndarray): label(s) for classification

        Returns:
            XGBClassifier: XGBoostClassifier
        """
        self.xgb_model.fit(X_train, y_train)
        return self.xgb_model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the model using common classification metrics.

        Args:
            X_test (np.ndarray): scaled features of the user(s)
            y_test (np.ndarray): label(s) for classification
        """
        y_pred = self.xgb_model.predict(X_test)
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]  # For probability scores (AUC-ROC)

        print("=== XGBoost Model Performance ===")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    def save_model(self, model_path: str = "model.joblib"):
        """Persist the trained model to disk using joblib.

        Args:
            model_path (str, optional): _description_. Defaults to "model.joblib".
        """
        joblib.dump(self.xgb_model, model_path)
        print(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str = "model.joblib"):
        """
        Load a saved model from disk using joblib.
        Returns an instance of Classifier with the loaded model.
        """
        instance = cls()  # Create a new instance
        instance.xgb_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return instance

    def predict(self, X: np.ndarray) -> List:
        """
        Use the loaded/trained model to make predictions.
        Expects X to be preprocessed (using the same preprocessor used during training).

        Args:
            X (np.ndarray): scaled features of the user(s)

        Returns:
            List: list of prediction
        """
        return self.xgb_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> List:
        """
        Get prediction probabilities for the positive class.

        Args:
            X (np.ndarray):  scaled features of the user(s)

        Returns:
            List: list of probabilities
        """
        return self.xgb_model.predict_proba(X)[:, 1]
