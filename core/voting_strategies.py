"""Patient-level voting strategies for aggregating slide predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from core.registry import register_voting


class BaseVotingStrategy(ABC):
    """Base class for patient-level aggregation strategies."""

    def __init__(self, **kwargs) -> None:
        self.params = kwargs

    @abstractmethod
    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        raise NotImplementedError


@register_voting('average')
class AverageVoting(BaseVotingStrategy):
    """Average probabilities per patient and threshold for prediction."""

    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        if num_classes == 1:
            grouped = results_df.groupby('patient_id')['prob_positive'].mean().reset_index()
            threshold = self.params.get('threshold', 0.5)
            grouped['prediction'] = (grouped['prob_positive'] > threshold).astype(int)
            return grouped

        prob_cols = [c for c in results_df.columns if c.startswith('prob_class_')]
        grouped = results_df.groupby('patient_id')[prob_cols].mean().reset_index()
        grouped['prediction'] = grouped[prob_cols].values.argmax(axis=1)
        return grouped


@register_voting('majority')
class MajorityVoting(BaseVotingStrategy):
    """Majority voting with probability-based tie breaking."""

    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        threshold = self.params.get('threshold', 0.5)
        records = []
        for patient_id, group in results_df.groupby('patient_id'):
            if num_classes == 1:
                vote_sum = group['prediction'].sum()
                total = len(group)
                if vote_sum > total / 2:
                    prediction = 1
                elif vote_sum < total / 2:
                    prediction = 0
                else:
                    prediction = int(group['prob_positive'].mean() > threshold)
                records.append(
                    {
                        'patient_id': patient_id,
                        'prediction': prediction,
                        'prob_positive': group['prob_positive'].mean(),
                    }
                )
                continue

            counts = group['prediction'].value_counts()
            top = counts[counts == counts.max()].index.tolist()
            if len(top) == 1:
                prediction = int(top[0])
            else:
                prob_cols = [c for c in group.columns if c.startswith('prob_class_')]
                summed = group[prob_cols].sum()
                prediction = int(summed.idxmax().replace('prob_class_', ''))

            record = {'patient_id': patient_id, 'prediction': prediction}
            prob_cols = [c for c in group.columns if c.startswith('prob_class_')]
            if prob_cols:
                record.update(group[prob_cols].mean().to_dict())
            records.append(record)

        return pd.DataFrame(records)
