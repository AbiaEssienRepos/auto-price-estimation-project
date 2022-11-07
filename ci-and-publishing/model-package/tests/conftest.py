import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_raw_dataset


@pytest.fixture()
def sample_input_data():

    # read dataset
    data = load_raw_dataset(file_name=config.app_config.test_data_file)
    data = data.drop(["car_ID", "enginelocation", "enginetype"], axis=1)

    return data
