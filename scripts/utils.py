import os
import numpy as np
import pandas as pd
from typing import List
from pipeline.tester import calc_accuracy, calc_precision_recall_f1


class DatasetDefaultConfig:
    use_pha = False
    segment_length_rate = 1
    overlap_length_rate = 0.75
    dataset_list = []

    # def __init__(self):
    #     '''ARIL'''
        # self.dataset_list.append("aril_none_%.2f_%.2f" % (self.segment_length_rate, self.overlap_length_rate))
        # '''WiFi_AR'''
        # for i in range(5):
        #     self.dataset_list.append("wifi_ar_%d_%.2f_%.2f" % (i, self.segment_length_rate, self.overlap_length_rate))
        # '''HTHI'''
        # for i in range(5):
        #     self.dataset_list.append("hthi_%d_none_%.2f_%.2f" % (i, self.segment_length_rate, self.overlap_length_rate))


def dataset_name_to_head_name_mapping(dataset_name: str):
    if dataset_name.startswith('aril'):
        return 'aril_span_cls'
    elif dataset_name.startswith('WiAR'):
        return 'wifi_ar_span_cls'
    elif dataset_name.startswith('hthi'):
        return 'hthi_span_cls'
    elif dataset_name.startswith('WiVio'):
        return 'wifi_ar_span_cls'


def break_down_dataset_name(whole_dataset_name: str):
    if whole_dataset_name.startswith('aril'):
        _, __, segment_length_rate, overlap_length_rate = whole_dataset_name.split("_")
        dataset_name = 'aril'
    elif whole_dataset_name.startswith('wifi_ar'):
        _, __, ___, segment_length_rate, overlap_length_rate = whole_dataset_name.split("_")
        dataset_name = 'wifi_ar'
    elif whole_dataset_name.startswith('hthi'):
        _, __, ___, segment_length_rate, overlap_length_rate = whole_dataset_name.split("_")
        dataset_name = 'hthi'
    else:
        return None, None, None
    segment_length_rate = float(segment_length_rate)
    overlap_length_rate = float(overlap_length_rate)
    return dataset_name, segment_length_rate, overlap_length_rate


def generate_result_summary(checkpoint_path: os.path, dataset_list: List):
    confusion_matrixs = {}
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}

    for dataset_name in dataset_list:
        dataset_path = os.path.join(checkpoint_path, dataset_name)

        strategies = os.listdir(dataset_path)
        for _strategy in strategies:
            file_list = os.listdir(os.path.join(dataset_path, _strategy))
            for file_name in file_list:
                if not file_name.endswith('confusion_matrix.csv'):
                    continue
                backbone_name, head_name, label_name, _ = file_name.split('-')
                _confusion_matrix = pd.read_csv(os.path.join(dataset_path, _strategy, file_name),
                                                sep=',', header=None)
                _dataset_name, segment_length_rate, overlap_length_rate = break_down_dataset_name(dataset_name)
                key = '%s-%.2f-%.2f-%s-%s-%s-%s' % (_dataset_name, segment_length_rate, overlap_length_rate,
                                                    backbone_name, head_name, _strategy, label_name)

                if key not in confusion_matrixs.keys():
                    confusion_matrixs[key] = _confusion_matrix.to_numpy()
                else:
                    confusion_matrixs[key] += _confusion_matrix.to_numpy()

    for key in confusion_matrixs.keys():
        _confusion_matrix = confusion_matrixs[key]
        n_classes = _confusion_matrix.shape[0]
        accuracy[key] = calc_accuracy(_confusion_matrix, n_classes)
        precision[key], recall[key], f1[key] = calc_precision_recall_f1(_confusion_matrix, n_classes)
        precision[key], recall[key], f1[key] = np.mean(precision[key]), np.mean(recall[key]), np.mean(f1[key])

    result = pd.DataFrame(columns=['dataset_name', 'segment_length_rate', 'overlap_length_rate',
                                   'backbone', 'head', 'strategy', 'label', 'accuracy', 'precision', 'recall', 'f1'])

    for key in confusion_matrixs.keys():
        dataset_name, segment_length_rate, overlap_length_rate, \
        backbone_name, head_name, _strategy, label_name = key.split('-')
        row = {'dataset_name': dataset_name,
               'segment_length_rate': segment_length_rate, 'overlap_length_rate': overlap_length_rate,
               'backbone': backbone_name, 'head': head_name, 'strategy': _strategy, 'label': label_name,
               'accuracy': accuracy[key], 'precision': precision[key], 'recall': recall[key], 'f1': f1[key]}
        result.loc[len(result)] = row
    result.to_csv(os.path.join('./result_summary.csv'), index=False)
