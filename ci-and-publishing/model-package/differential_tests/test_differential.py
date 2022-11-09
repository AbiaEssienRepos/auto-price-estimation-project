import math

import pandas as pd

from regression_model.config.core import DATASET_DIR, config
from regression_model.predict import make_prediction
from regression_model.processing.data_manager import load_raw_dataset


def test_model_prediction_differential(*, save_file: str = "test_predictions.csv"):
    """
    This test compares the prediction result similarity of
    the current model with the previous model's results.
    """

    # Given
    # Load the saved previous model predictions
    previous_model_df = pd.read_csv(f"{DATASET_DIR}/{save_file}")
    previous_model_predictions = previous_model_df.values

    test_data = load_raw_dataset(file_name=config.app_config.test_data_file)
    test_data = test_data.drop(["car_ID", "enginelocation", "enginetype"], axis=1)

    # When
    current_result = make_prediction(input_data=test_data)
    current_model_predictions = current_result.get("predictions")

    # Then
    # diff the current model vs. the old model
    assert len(previous_model_predictions) == len(current_model_predictions)

    # Perform the differential test
    for previous_value, current_value in zip(
        previous_model_predictions, current_model_predictions
    ):

        # convert numpy float64 to Python float.
        previous_value = previous_value.item()
        current_value = current_value.item()

        # rel_tol is the relative tolerance – it is the maximum allowed
        # difference between a and b, relative to the larger absolute
        # value of a or b. For example, to set a tolerance of 5%, pass
        # rel_tol=0.05.
        assert math.isclose(previous_value, current_value, rel_tol=0.05)
