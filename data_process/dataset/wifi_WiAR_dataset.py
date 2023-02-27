import os
import logging
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from data_process.dataset_config import DatasetConfig
from util import log_f_ch

logger = logging.getLogger(__name__)

class WiFiARDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path, train_val_rate: float):
        super(WiFiARDatasetConfig, self).__init__('WiAR', datasource_path)
        self.train_val_rate = train_val_rate


def load_WiAR_dataset(config: WiFiARDatasetConfig):

    logger.info(f'加载 {config.dataset_name} 数据集')

    full_dataset = torch.load(config.datasource_path)

    train_size = int(config.train_val_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size

    logger.info(log_f_ch('训练数据集长度：', str(train_size)))
    logger.info(log_f_ch('测试数据集长度：', str(test_size)))

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset


class WiFiARDataset(Dataset):
    def __init__(self, dataset):
        super(WiFiARDataset, self).__init__()
        logger.info(f'加载 WiAR 数据集')
        self.dataset = dataset
        self.label_n_class = 7

        self.num_sample = len(dataset)
        amp, label = next(iter(dataset))
        self.seq_len, self.n_channel = amp.shape

        _, self.freq_n_channel, self.freq_seq_len = None, None, None

        logger.info(log_f_ch('num_sample: ', str(self.num_sample)))
        logger.info(log_f_ch('n_class: ', str(self.label_n_class)))
        logger.info(log_f_ch('seq_len: ', str(self.seq_len)))
        logger.info(log_f_ch('n_channel: ', str(self.n_channel)))


    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = data.transpose(1, 0)

        return {
            'data': data.float(),
            # 'freq_data': None,
            'label': label.long(),
        }

    def __len__(self):
        return self.num_sample

    def get_n_channels(self):
        return {
            'data': self.n_channel,
            'freq_data': self.freq_n_channel,
        }

    def get_seq_lens(self):
        return {
            'data': self.seq_len,
            'freq_data': self.freq_seq_len,
        }

    def get_n_classes(self):
        return {
            'label': self.label_n_class,
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')
    datasource_path = os.path.join("D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi\dataset\WiAR\Data.pt")
    train_val_rate = 0.8
    dataset_config = WiFiARDatasetConfig(datasource_path,train_val_rate)
    train_dataset, test_dataset = load_WiAR_dataset(dataset_config)
    train_dataset, test_dataset = WiFiARDataset(train_dataset), WiFiARDataset(test_dataset)

    print(len(train_dataset))
    from torch.utils.data.dataloader import DataLoader
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                     drop_last=True)
    for i, data in enumerate(loader):
        print(i)
    # print(train_dataset[0]['data'].shape)