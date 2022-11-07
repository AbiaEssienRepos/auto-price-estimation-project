from config.core import config
from pipeline import price_pipe
from processing.data_manager import load_raw_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_raw_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        # independent variables
        data.drop(config.model_config.dropped_in_split, axis=1),
        # target variable
        data[config.model_config.target],
        # test size (0.3 in config)
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # fit model
    price_pipe.fit(X_train, y_train)
    pred = price_pipe.predict(X_test)
    print(pred[0])

    # persist trained model
    save_pipeline(pipeline_to_persist=price_pipe)


if __name__ == "__main__":
    run_training()
