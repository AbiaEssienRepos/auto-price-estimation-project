# from scikit-learn
from feature_engine.encoding import RareLabelEncoder

# from feature engine
from feature_engine.selection import DropFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

# from config
from regression_model.config.core import config

# created in-house
from regression_model.processing import preprocessors as pp

price_pipe = Pipeline(
    [
        # ===== REMAP SYMBOLING =====
        # reassign the value to reflect domain reality
        (
            "remap_symboling",
            pp.RemapVariable(
                variables=config.model_config.remap_vars,
            ),
        ),
        # ===== CAR NAME =====
        # collapse the cardinality of the column
        (
            "collapse_carname",
            pp.CarTransform(
                variable=config.model_config.car_var,
            ),
        ),
        # ===== RARE LABELS =====
        # group categories present in less than 1% of observations
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.01, n_categories=1, variables=config.model_config.cat_vars
            ),
        ),
        # ===== CATEGORICAL ENCODER =====
        # one-hot encode binary variables
        (
            "binary_encoder",
            pp.CategoricalEncoder(
                variables=config.model_config.binary_vars,
                target=config.model_config.target,
            ),
        ),
        # encoding of non-binary variables
        (
            "non_binary_encoder",
            pp.OrdinalEncoder(
                variables=config.model_config.non_binary_vars,
                target=config.model_config.target,
            ),
        ),
        # ==== SCALER =====
        # scale the continuous variables
        ("scaler", pp.ContinuousScaler(variables=config.model_config.scaled_features)),
        # === DROP FEATURES ===
        # reduce dataset to selected features
        (
            "drop_features",
            DropFeatures(features_to_drop=config.model_config.dropped_features),
        ),
        # ===== TRAIN MODEL =====
        (
            "lasso",
            Lasso(
                alpha=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
