from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data = validated_data.dropna(subset=new_vars_with_na)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data.copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleCarDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class CarDataInputSchema(BaseModel):
    car_ID: Optional[int]
    symboling: Optional[int]
    CarName: Optional[str]
    fueltype: Optional[str]
    aspiration: Optional[str]
    doornumber: Optional[str]
    carbody: Optional[str]
    drivewheel: Optional[str]
    enginelocation: Optional[str]
    wheelbase: Optional[float]
    carlength: Optional[float]
    carwidth: Optional[float]
    carheight: Optional[int]
    curbweight: Optional[int]
    enginetype: Optional[str]
    cylindernumber: Optional[str]
    enginesize: Optional[int]
    fuelsystem: Optional[str]
    boreratio: Optional[float]
    stroke: Optional[float]
    compressionratio: Optional[float]
    horsepower: Optional[int]
    peakrpm: Optional[int]
    citympg: Optional[int]
    highwaympg: Optional[int]


class MultipleCarDataInputs(BaseModel):
    inputs: List[CarDataInputSchema]
