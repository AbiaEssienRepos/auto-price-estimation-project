from typing import Any, List, Optional

from pydantic import BaseModel
from regression_model.processing.validation import CarDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleCarDataInputs(BaseModel):
    inputs: List[CarDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "car_ID": 1,
                        "symboling": 3,
                        "CarName": "alfa-romeo",
                        "fueltype": "gas",
                        "aspiration": "std",
                        "doornumber": "two",
                        "carbody": "convertible",
                        "drivewheel": "rwd",
                        "enginelocation": "front",
                        "wheelbase": 88.6,
                        "carlength": 168.8,
                        "carwidth": 64.1,
                        "carheight": 48.8,
                        "curbweight": 2548,
                        "enginetype": "dohc",
                        "cylindernumber": "four",
                        "enginesize": 130,
                        "fuelsystem": "mpfi",
                        "boreratio": 3.47,
                        "stroke": 2.68,
                        "compressionratio": 9.0,
                        "horsepower": 111,
                        "peakrpm": 5000,
                        "citympg": 21,
                        "highwaympg": 27,
                    }
                ]
            }
        }
