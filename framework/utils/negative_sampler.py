from framework.preprocessing.signals import SignalSource
"""Utilities to to generate or modify data samples."""

import numpy as np
import pandas as pd



def get_neg_sample(pos_sample: pd.DataFrame,
                   n_points: int,
                   do_permute: bool = False,
                   delta: float = 0.0) -> pd.DataFrame:
    """Creates a negative sample from the cuboid bounded by +/- delta.
        Where, [min - delta, max + delta] for each of the dimensions.
        If do_permute, then rather than uniformly sampling, simply
        randomly permute each dimension independently.
        The positive sample, pos_sample is a pandas DF that has a column
        labeled 'class_label' where 1.0 indicates Normal, and
        0.0 indicates anomalous.
        Args:
          pos_sample: DF with numeric dimensions
          n_points: number points to be returned
          do_permute: permute or sample
          delta: fraction of [max - min] to extend the sampling.
        Returns:
          A dataframe  with the same number of columns, and a label column
          'class_label' where every point is 0.
    """
    df_neg = pd.DataFrame()
    
    pos_sample_n = pos_sample.sample(n=n_points, replace=True)
    
    for field_name in list(pos_sample):
    
        if field_name == "class_label":
            continue
    
        if do_permute:
            df_neg[field_name] = np.random.permutation(
            np.array(pos_sample_n[field_name]))
    
        else:
            low_val = min(pos_sample[field_name])
            high_val = max(pos_sample[field_name])
            delta_val = high_val - low_val
            df_neg[field_name] = np.random.uniform(
                low=low_val - delta * delta_val,
                high=high_val + delta * delta_val,
                size=n_points)

    df_neg["class_label"] = [1 for _ in range(n_points)]
    return df_neg


def apply_negative_sample(positive_sample: pd.DataFrame, sample_ratio: float,
                          sample_delta: float) -> pd.DataFrame:
    """Returns a dataset with negative and positive sample.
    Args:
      positive_sample: actual, observed sample where each col is a feature.
      sample_ratio: the desired ratio of negative to positive points
      sample_delta: the extension beyond observed limits to bound the neg sample
    Returns:
      DataFrame with features + class label, with 1 being observed and 0 negative.
    """

    positive_sample["class_label"] = 1
    n_neg_points = int(len(positive_sample) * sample_ratio)
    negative_sample = get_neg_sample(
        positive_sample, n_neg_points, do_permute=False, delta=sample_delta)
    training_sample = pd.concat([positive_sample, negative_sample],
                                ignore_index=True,
                                sort=True)
    return training_sample.reindex(np.random.permutation(training_sample.index))

def apply_negative_samples(pos_df, signals, sample_ratio, sample_delta):
    n_neg_points = int(len(pos_df) * sample_ratio)
    n_neg_points += 1
    df_neg = pos_df.copy()
    df_neg["class_label"] = 1
    
    tempt_df = pos_df.sample(n=n_neg_points, replace=True)
    
    neg_samples = tempt_df.index.values.tolist()
    
    for signal in signals:
    
        if signal.source != SignalSource.sensor:
            continue
    
        low_val = signal.min_value
        high_val = signal.max_value
        delta_val = high_val - low_val
        df_neg.loc[neg_samples,signal.name] = np.random.uniform(
            low=low_val - sample_delta * delta_val,
            high=high_val + sample_delta * delta_val,
            size=n_neg_points)

    df_neg.loc[neg_samples,"class_label"] = 0
    return df_neg


