import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.preprocessor = None

    @classmethod
    def from_csv(cls, filepath: str):
        data = pd.read_csv(filepath)
        return cls(data)

    def preprocess_data(
        self,
        target_col: str = "completed_filing",
        test_size: float = 0.2,
        inference_size: float = 0.5,
        output_dir: str = "data",
        random_state: int = 42,
    ):
        """
        Splits the dataset into training, testing, and inference sets, then saves them to CSV files.

        Args:
            target_col (str): The name of the target column.
            test_size (float): Proportion of data to be used for testing + inference (default: 20%).
            inference_size (float): Proportion of the test set used for inference\
                (default: 50% of test set).
            output_dir (str): Directory to save the split datasets (default: "data").
        """
        if self.data is None:
            raise ValueError("No data available for preprocessing. Please load data first.")

        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)

        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]

        # Define categorical and numerical features
        categorical_features = [
            "employment_type",
            "marital_status",
            "device_type",
            "referral_source",
        ]
        numerical_features = [
            "age",
            "income",
            "time_spent_on_platform",
            "number_of_sessions",
            "fields_filled_percentage",
            "previous_year_filing",
        ]

        # Create pipelines for preprocessing
        self.preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        # Fit and transform the features
        X_transformed = self.preprocessor.fit_transform(X)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state
        )

        # Second split: 10% Testing, 10% Inference
        X_test, X_inference, y_test, y_inference = train_test_split(
            X_test, y_test, test_size=inference_size, random_state=random_state, stratify=y_test
        )

        return X_train, X_test, y_train, y_test, self.preprocessor

    def _save_dataset(self, X, y, columns, target: str, output_dir: str, data_type: str):

        X_df = pd.DataFrame(X, columns=columns)
        X_df.to_csv(f"{output_dir}/{data_type}_features.csv", index=False)
        if y:
            y_df = pd.DataFrame(y, columns=[target])
            y_df.to_csv(f"{output_dir}/{data_type}_labels.csv", index=False)

    def save_preprocessor(self, filepath: str = "preprocessor.joblib"):
        """Save the preprocessor object to disk.

        Args:
            filepath (str, optional): file path to store preprocessor object.\
                Defaults to "preprocessor.joblib".
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor has not been fitted yet. Please run preprocess_data first."
            )
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath: str = "preprocessor.joblib"):
        """
        Create a Preprocessor instance with a loaded preprocessor.
        Note: This method does not require a DataFrame, as it's intended for inference.

        Args:
            filepath (str, optional): _description_. Defaults to "preprocessor.joblib".

        Returns:
            _type_: _description_
        """
        instance = cls()
        instance.preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return instance

    def transform_new_data(self, new_data: pd.DataFrame):
        """
        Transform new data using the loaded/fitted preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor has not been set. Please load or fit the preprocessor first."
            )
        return self.preprocessor.transform(new_data)
