def test_schema(sample_input_data):

    # Given
    expected_no_of_columns = 22

    # When
    data = sample_input_data
    actual_columns = len(data.columns)

    # Then
    assert actual_columns == expected_no_of_columns


def test_missing_values(sample_input_data):

    # Given
    expected_vars_with_na = 0

    # When
    actual_vars_with_na = [
        var
        for var in sample_input_data.columns
        if sample_input_data[var].isnull().sum() > 0
    ]

    # Then
    assert len(actual_vars_with_na) == expected_vars_with_na
