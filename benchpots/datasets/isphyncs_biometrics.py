"""
Preprocessing func for the dataset iSPHYNCS Biometrics.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>, Wenjie Du <wenjay.du@gmail.com>, and Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

import numpy as np
import tsdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness

ORDERED_FEATURES = [
    'heart',
    'steps',
    'calories',
    'sleep_stage',
]


def preprocess_isphyncs_biometrics(
    n_steps: int,
    rate: float,
    pattern: str = "point",
    features: list = None,
    **kwargs,
) -> dict:
    """Load and preprocess the dataset iSPHYNCS Biometrics.

    Parameters
    ----------
    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    features:
        The features to be used in the dataset.
        If None, all features except the static features will be used.

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed iSPHYNCS Biometrics.

    """

    def truncate(df_temp, min_trunc_length):  # pad and truncate to set the max length
        if len(df_temp) < min_trunc_length:
            return None
        else:
            n_truncs = len(df_temp) // min_trunc_length
            df_temp = df_temp.iloc[:n_truncs*min_trunc_length]  # truncate
        return df_temp

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"

    # read the raw data
    data = tsdb.load("isphyncs_biometrics")
    all_features = set(data["test_X"].columns)
    main_feature = "heart"  # feature "heart" is the main feature to indicate that the device was worn
    time_feature = "time"  # date time feature
    train_X = data['train_X'].reset_index(drop=True)
    test_X = data['test_X'].reset_index(drop=True)

    if features is not None:  # if features are specified by users, only use the specified features
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
        if main_feature not in features:
            features.append(main_feature)
        if time_feature not in features:
            features.append(time_feature)
    else:
        features = list(all_features)
        
    # select the specified features finally
    train_X = train_X[features]
    test_X = test_X[features]

    if n_steps is None:
        n_steps = np.min([train_X.groupby('RecordID').size().min(),
                          test_X.groupby('RecordID').size().min()])
    else:
        assert n_steps > 0, f"n_steps must be larger than 0, but got {n_steps}"
        min_n_steps = np.min([train_X.groupby('RecordID').size().min(),
                            test_X.groupby('RecordID').size().min()])
        assert n_steps <= min_n_steps, f"n_steps must be smaller than the minimum number of steps in the dataset ({min_n_steps}), but got {n_steps}"
    train_X = train_X.groupby("RecordID").apply(truncate, n_steps)
    train_X = train_X.drop("RecordID", axis=1)
    train_X = train_X.reset_index()
    train_X = train_X.drop(["level_1"], axis=1)
    test_X = test_X.groupby("RecordID").apply(truncate, n_steps)
    test_X = test_X.drop("RecordID", axis=1)
    test_X = test_X.reset_index()
    test_X = test_X.drop(["level_1"], axis=1)

    # split the dataset into the train, val, and test sets
    train_recordID = train_X["RecordID"].unique()
    test_set_ids = test_X["RecordID"].unique()
    train_set_ids, val_set_ids = train_test_split(train_recordID, test_size=0.2)
    train_set_ids.sort(), val_set_ids.sort()
    val_X = train_X[train_X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", time_feature])
    train_X = train_X[train_X["RecordID"].isin(train_set_ids)].sort_values(["RecordID", time_feature])
    test_X = test_X.sort_values(["RecordID", time_feature])
    
    # extract the source (RecordID)
    train_X_source = train_X["RecordID"].to_numpy().reshape(-1, n_steps)[:, 0]
    val_X_source = val_X["RecordID"].to_numpy().reshape(-1, n_steps)[:, 0]
    test_X_source = test_X["RecordID"].to_numpy().reshape(-1, n_steps)[:, 0]

    # remove useless columns, sort, and turn into numpy arrays
    train_X = train_X.drop(["RecordID", time_feature], axis=1)
    val_X = val_X.drop(["RecordID", time_feature], axis=1)
    test_X = test_X.drop(["RecordID", time_feature], axis=1)
    train_X = train_X[ORDERED_FEATURES]
    val_X = val_X[ORDERED_FEATURES]
    test_X = test_X[ORDERED_FEATURES]
    train_X, val_X, test_X = (
        train_X.to_numpy(),
        val_X.to_numpy(),
        test_X.to_numpy(),
    )

    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # reshape into time series samples
    train_X = train_X.reshape(-1, n_steps, train_X.shape[-1])
    val_X = val_X.reshape(-1, n_steps, train_X.shape[-1])
    test_X = test_X.reshape(-1, n_steps, train_X.shape[-1])

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        "train_X_source": train_X_source,
        # val set
        "val_X": val_X,
        "val_X_source": val_X_source,
        # test set
        "test_X": test_X,
        "test_X_source": test_X_source,
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
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset