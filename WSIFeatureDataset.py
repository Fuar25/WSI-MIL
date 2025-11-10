from typing import Optional
from torch import tensor as torch_tensor
from torch import float32
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

class NPYDataset(Dataset):
    def __init__(self, root, csv_path, score: Optional[bool] = False):
        self.root = root
        self.features_path_list = []
        self.patho_id_list = []
        self.label_list = []
        self.csv = pd.read_csv(csv_path)
        self._get_all_path()
        self._get_all_patho_id()
        self._get_all_label()
        if score:
            self.score_list = self._get_all_scores()
        else:
            self.score_list = None

    def _get_all_path(self):
        for features_name in os.listdir(self.root):
            features_path = os.path.join(self.root, features_name)
            self.features_path_list.append(features_path)

    def _get_all_patho_id(self):
        for feature_path in self.features_path_list:
            patho_id = f"{os.path.basename(feature_path).split('_ctranspath')[0]}"
            self.patho_id_list.append(patho_id)

    def _get_all_label(self):
        new_patho_id_list = []
        new_features_path_list = []
        new_label_list = []

        for patho_id, features_path in zip(self.patho_id_list, self.features_path_list):
            index = self.csv['file_name'] == patho_id
            if index.any():
                label = self.csv[index]['GT'].iloc[0]
                new_label_list.append(label)
                new_patho_id_list.append(patho_id)
                new_features_path_list.append(features_path)

        self.patho_id_list = new_patho_id_list
        self.label_list = new_label_list
        self.features_path_list = new_features_path_list


    def _get_all_scores(self):
        new_score_list = []
        new_patho_id_list = []
        new_features_path_list = []
        new_label_list = []

        for patho_id, features_path, label in zip(self.patho_id_list, self.features_path_list, self.label_list):
            index = self.csv['file_name'] == patho_id
            if index.any():
                score = self.csv[index]['SCORE'].iloc[0]
                new_score_list.append(score)
                new_patho_id_list.append(patho_id)
                new_features_path_list.append(features_path)
                new_label_list.append(label)

        self.patho_id_list = new_patho_id_list
        self.label_list = new_label_list
        self.features_path_list = new_features_path_list
        return new_score_list

    def __len__(self):
        return len(self.features_path_list)

    def __getitem__(self, idx):
        features = torch_tensor(np.load(self.features_path_list[idx]), dtype=float32)
        label = torch_tensor([self.label_list[idx]], dtype=float32)
        if self.score_list is not None :
            score = torch_tensor([self.score_list[idx]], dtype=float32)
            return features, label, score
        return features, label

# To control the situation where treats 00 and 01 as two category
class NPYDataset_with_dirname(Dataset):
    def __init__(self, root, csv_path, score: Optional[bool] = False):
        self.root = root
        self.features_path_list = []
        self.patho_id_list = []
        self.label_list = []
        self.csv = pd.read_csv(csv_path)
        self._get_all_path()
        self._get_all_patho_id()
        self._get_all_label()
        if score:
            self.score_list = self._get_all_scores()
        else:
            self.score_list = None

    def _get_all_path(self):
        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            for features_name in os.listdir(class_path):
                features_path = os.path.join(class_path, features_name)
                self.features_path_list.append(features_path)

    def _get_all_patho_id(self):
        for features_path in self.features_path_list:
            patho_id = f"{os.path.basename(features_path).split('_ctranspath')[0]}"
            self.patho_id_list.append(patho_id)

    def _get_all_label(self):
        label_name_list = os.listdir(self.root)
        for features_path in self.features_path_list:
            label_name = os.path.basename(os.path.dirname(features_path))
            label = label_name_list.index(label_name)
            self.label_list.append(label)

    def _get_all_scores(self):
        new_score_list = []
        new_patho_id_list = []
        new_features_path_list = []
        new_label_list = []

        for patho_id, features_path, label in zip(self.patho_id_list, self.features_path_list, self.label_list):
            index = self.csv['file_name'] == patho_id
            if index.any():
                score = self.csv[index]['SCORE'].iloc[0]
                new_score_list.append(score)
                new_patho_id_list.append(patho_id)
                new_features_path_list.append(features_path)
                new_label_list.append(label)

        self.patho_id_list = new_patho_id_list
        self.label_list = new_label_list
        self.features_path_list = new_features_path_list
        return new_score_list

    def __len__(self):
        return len(self.features_path_list)

    def __getitem__(self, idx):
        features = torch_tensor(np.load(self.features_path_list[idx]), dtype=float32)
        label = torch_tensor([self.label_list[idx]], dtype=float32)
        if self.score_list is not None :
            score = torch_tensor([self.score_list[idx]], dtype=float32)
            return features, label, score
        return features, label

if __name__ == '__main__':
    root = '/mnt/gml/PD-L1/previous/features/cpath_feature'
    csv_path = '/home/william/Desktop/gml/ALL_with_score.csv'
    control_dataset = NPYDataset_with_dirname(root=root, csv_path=csv_path, score=True)
    print()