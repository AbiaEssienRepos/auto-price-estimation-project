import numpy as np

from regression_model.predict import make_prediction

# from regression_model.processing.data_manager import save_test_predictions


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 7217
    expected_no_predictions = 62

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert round(predictions[0]) == expected_first_prediction_value

    # save the predictions
    # save_test_predictions(pred_to_persist=predictions)
