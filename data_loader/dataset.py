
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from core.base import BaseDataset
from core.registry import register_dataset


def _extract_patient_id(filename: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


@register_dataset('wsi_h5')
class WSIDataset(BaseDataset):
    """Dataset for loading WSI features and coords from H5 files."""

    def __init__(
        self,
        data_paths: Dict[str, str],
        feature_key: str = 'features',
        coords_key: str = 'coords',
        patient_id_pattern: str = r"((?:xs)?B\d{4}-\d{5})",
        binary_mode: Optional[bool] = None,
    ) -> None:
    
        self.data_paths = data_paths
        self.feature_key = feature_key
        self.coords_key = coords_key
        self.patient_id_pattern = patient_id_pattern
        self.classes = sorted(data_paths.keys())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

        self.samples: List[Dict[str, Any]] = []
        self._scan_files()
        self.data = self.samples  # backward compatibility

        print(
            f"WSI Dataset loaded: {len(self.samples)} samples across {len(self.classes)} classes"
        )
        print(f"Class mapping: {self.class_to_idx}")
        print(f"Mode: {'Binary' if self.binary_mode else 'Multi-class'}")

    def _scan_files(self) -> None:
        for cls_name, dir_path in self.data_paths.items():
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} for class {cls_name} does not exist.")
                continue

            label = self.class_to_idx[cls_name]
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith('.h5'):
                        continue
                    patient_id = _extract_patient_id(filename, self.patient_id_pattern)
                    if patient_id is None:
                        print(f"Warning: Could not extract Patient ID from {filename}. Skipping this file.")
                        continue
                    self.samples.append(
                        {
                            'sample_id': filename.replace('.h5', ''),
                            'patient_id': patient_id,
                            'filename': filename,
                            'feature_path': os.path.join(root, filename),
                            'label': label,
                            'class_name': cls_name,
                        }
                    )

    def get_patient_ids(self) -> List[str]:
        return [item['patient_id'] for item in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        with h5py.File(item['feature_path'], 'r') as f:
            features = torch.from_numpy(np.array(f[self.feature_key])).float()
            coords = (
                torch.from_numpy(np.array(f[self.coords_key])).float()
                if self.coords_key in f
                else torch.zeros(features.shape[0], 2)
            )

        label_tensor = (
            torch.tensor(item['label']).float()
            if self.binary_mode
            else torch.tensor(item['label']).long()
        )

        return {
            'features': features,  # (N, C)
            'coords': coords,  # (N, 2)
            'label': label_tensor,
            'sample_id': item['sample_id'],
            'patient_id': item['patient_id'],
        }

