"""
The preprocessing function of the dataset PeMS traffic.

Refer to https://pems.dot.ca.gov
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pandas as pd
from pypots.data import sliding_window
from pypots.utils.logging import logger
from sklearn.preprocessing import StandardScaler

from .utils import create_missingness, print_final_dataset_info


def preprocess_pems_traffic(
    file_path,
    rate,
    n_steps,
    pattern: str = "point",
    **kwargs,
):
    """
    Returns
    -------
    data : dict
        A dictionary contains X:
            X : pandas.DataFrame
                The time-series data of Electricity Load Diagrams.
    """

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    df = pd.read_csv(file_path)

    feature_names = df.columns.tolist()
    feature_names.remove("date")
    df["date"] = pd.to_datetime(df["date"])

    unique_months = df["date"].dt.to_period("M").unique()
    selected_as_train = unique_months[:15]  # use the first 15 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[15:19]  # select the following 4 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[
        19:
    ]  # select the left 6 months as test set, 2018-07 has only 2 days, so can be rounded to 5 months
    logger.info(f"months selected as test set are {selected_as_test}")

    test_set = df[df["date"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_X = scaler.transform(val_set.loc[:, feature_names])
    test_X = scaler.transform(test_set.loc[:, feature_names])

    train_X = sliding_window(train_X, n_steps)
    val_X = sliding_window(val_X, n_steps)
    test_X = sliding_window(test_X, n_steps)

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
    }

    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = train_X
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

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
