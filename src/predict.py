import numpy as np
import pandas as pd

from config import paths
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from prediction.data_loader import load_data_loader_factory
from utils import (
    save_dataframe_as_csv,
    ResourceTracker,
)

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
    ids: np.ndarray, probs: np.ndarray, predictions: np.ndarray, class_to_idx: dict
) -> pd.DataFrame:
    """
    Creates a DataFrame containing predictions and their associated probabilities for each class.

    Args:
    - ids (np.ndarray): An array of identifiers for the samples.
    - probs (np.ndarray): A 2D array where each row contains the probabilities for each class for a given sample.
    - predictions (np.ndarray): An array of class indices predicted for each sample.
    - class_to_idx (dict): A dictionary mapping class names to their respective indices.

    Returns:
    - pd.DataFrame: A DataFrame with the following columns:
        - 'id': The identifier for each sample.
        - One column for each class, containing the probability of that class for each sample. The columns are named after the class names.
        - 'prediction': The predicted class name for each sample.
    """
    idx_to_class = {k: v for v, k in class_to_idx.items()}
    encoded_targets = list(range(len(class_to_idx)))
    prediction_df = pd.DataFrame({"id": ids})
    prediction_df[encoded_targets] = probs
    prediction_df["prediction"] = predictions
    prediction_df["prediction"] = prediction_df["prediction"].map(idx_to_class)
    prediction_df.rename(columns=idx_to_class, inplace=True)
    return prediction_df


def run_batch_predictions(
    test_dir_path: str = paths.TEST_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
    data_loader_file_path: str = paths.SAVED_DATA_LOADER_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    Args:
        test_dir_path (str): Directory path for the test data.
        predictor_dir_path (str): Path to the directory of saved model.
        predictions_file_path (str): Path where the predictions file will be saved.
        data_loader_file_path (str): Path to the saved data loader file.
    """

    try:
        with ResourceTracker(logger, monitoring_interval=5) as _:
            logger.info("Making batch predictions...")

            logger.info("Loading test data...")
            data_loader_factory = load_data_loader_factory(data_loader_file_path)
            test_data, image_names = data_loader_factory.create_test_data_loader(
                data_dir_path=test_dir_path,
            )

            logger.info("Loading predictor model...")
            predictor_model = load_predictor_model(predictor_dir_path)

            logger.info("Making predictions...")
            predicted_labels, predicted_probabilities = predict_with_model(
                predictor_model, test_data
            )

            logger.info("Creating final predictions dataframe...")
            predictions_df = create_predictions_dataframe(
                ids=image_names,
                probs=predicted_probabilities,
                predictions=predicted_labels,
                class_to_idx=data_loader_factory.class_to_idx,
            )

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
