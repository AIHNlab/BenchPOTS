"""
The preprocessing function of the dataset PhysionNet2019 for BenchPOTS.
dataset path https://physionet.org/content/challenge-2019/1.0.0/
"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pandas as pd
import tsdb
from pypots.utils.logging import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import create_missingness, print_final_dataset_info


def preprocess_physionet2019(
    rate,
    pattern: str = "point",
    subset="all",
    features: list = None,
    **kwargs,
):
    """Generate a fully-prepared PhysioNet2019 dataset for benchmarking and validating POTS models.

    Parameters
    ----------
    rate: float,
        The additional missing rate to artificially add to the dataset.
        If the dataset has original missing values, this rate won't be applied to them.
        If the dataset originally has no missing data, this rate will be applied to the dataset.

    pattern

    subset

    features

    Returns
    -------
    processed_dataset: dict,
        A dictionary containing the processed PhysioNet-2019 dataset.

    """

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        if len(df_temp) < 48:
            return None
        else:
            df_temp = df_temp.set_index("ICULOS").sort_index().reset_index()
            df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    all_subsets = ["all", "training_setA", "training_setB"]
    assert (
        subset in all_subsets
    ), f"subset should be one of {all_subsets}, but got {subset}"
    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"

    # read the raw data
    data = tsdb.load("physionet_2019")
    all_features = set(data["training_setA"].columns)
    label_feature = "SepsisLabel"  # feature SepsisLabel contains labels indicating whether patients get sepsis
    time_feature = "ICULOS"  # ICU length-of-stay (hours since ICU admit)

    if subset != "all":
        df = data[subset]
        X = df.reset_index(drop=True)
    else:
        df = pd.concat([data["training_setA"], data["training_setB"]], sort=True)
        X = df.reset_index(drop=True)

    if (
        features is None
    ):  # if features are not specified, we use all features except the static features, e.g. age
        X = X.drop(data["static_features"], axis=1)
    else:  # if features are specified by users, only use the specified features
        # check if the given features are valid
        features_set = set(features)
        if not all_features.issuperset(features_set):
            intersection_feats = all_features.intersection(features_set)
            difference = features_set.difference(intersection_feats)
            raise ValueError(
                f"Given features contain invalid features that not in the dataset: {difference}"
            )
        # check if the given features contain necessary features for preprocessing
        if "RecordID" not in features:
            features.append("RecordID")
        if label_feature not in features:
            features.append(label_feature)
        if time_feature not in features:
            features.append(time_feature)

        # select the specified features finally
        X = X[features]

    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)
    X = X.reset_index()
    X = X.drop(["level_1"], axis=1)

    # split the dataset into the train, val, and test sets
    all_recordID = X["RecordID"].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)

    train_set_ids.sort()
    val_set_ids.sort()
    test_set_ids.sort()
    train_set = X[X["RecordID"].isin(train_set_ids)].sort_values(
        ["RecordID", time_feature]
    )
    val_set = X[X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", time_feature])
    test_set = X[X["RecordID"].isin(test_set_ids)].sort_values(
        ["RecordID", time_feature]
    )
    train_y = train_set[[time_feature, label_feature]]
    val_y = val_set[[time_feature, label_feature]]
    test_y = test_set[[time_feature, label_feature]]

    # remove useless columns and turn into numpy arrays
    train_set = train_set.drop(["RecordID", time_feature, label_feature], axis=1)
    val_set = val_set.drop(["RecordID", time_feature, label_feature], axis=1)
    test_set = test_set.drop(["RecordID", time_feature, label_feature], axis=1)
    train_X, val_X, test_X = (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
    )

    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # reshape into time series samples
    train_X = train_X.reshape(len(train_set_ids), 48, -1)
    val_X = val_X.reshape(len(val_set_ids), 48, -1)
    test_X = test_X.reshape(len(test_set_ids), 48, -1)

    # fetch labels for train/val/test sets
    train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_classes": 2,
        "n_steps": 48,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        "train_y": train_y,
        # val set
        "val_X": val_X,
        "val_y": val_y,
        # test set
        "test_X": test_X,
        "test_y": test_y,
    }

    if rate > 0:
        logger.warning(
            "Note that physionet_2019 has sparse observations in the time series, "
            "hence we don't add additional missing values to the training dataset. "
        )

        # hold out ground truth in the original data for evaluation
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        # test_X_ori is for error calc, not for model input, hence mustn't have NaNs
        processed_dataset["test_X_ori"] = np.nan_to_num(
            test_X_ori
        )  # fill NaNs for later error calc
        processed_dataset["test_X_indicating_mask"] = np.isnan(test_X_ori) ^ np.isnan(
            test_X
        )
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
