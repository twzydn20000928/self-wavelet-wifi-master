import os
import logging
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from data_process.dataset_config import DatasetConfig

from util import log_f_ch, load_mat

logger = logging.getLogger( )


class WiFiVioDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path, save_path: os.path):
        super(WiFiVioDatasetConfig, self).__init__('wifiVio', datasource_path
                                                   , save_path)

def load_wifi_Vio_data(config: WiFiVioDatasetConfig):

    train_data = {"data_path": os.path.join(config.datasource_path, 'train'),
                  "list_path": os.path.join(config.datasource_path, 'train_list.csv'),
                  "save_path": os.path.join(config.save_path, 'Train_dataset','train_dataset.csv')}
    test_data = {"data_path": os.path.join(config.datasource_path, 'test'),
                  "list_path": os.path.join(config.datasource_path, 'test_list.csv'),
                 "save_path": os.path.join(config.save_path, 'Test_dataset', 'test_dataset.csv')}

    for data in [train_data, test_data]:
        if os.path.exists(data['save_path']):
            os.remove(data['save_path'])

    return train_data, test_data

class WiFiVioDataset(Dataset):
    def __init__(self, data, is_test: bool):
        super(WiFiVioDataset, self).__init__()

        logger.info('加载WiFiVio数据集')
        self.is_test = is_test
        self.data_path = data['data_path']
        self.data_list = pd.read_csv(data['list_path'])
        self.save_path = data['save_path']
        self.tmp_data = self.data_list.iloc[0]['file']
        self.tmp_data = load_mat(os.path.join(data['data_path'], f'{self.tmp_data}.h5'))

        self.n_channel, self.seq_len = self.tmp_data['amp'].shape
        self.num_sample = len(self.data_list)

        self.label_n_class = 7
        self.freq_n_channel, self.freq_seq_len = None, None

        logger.info(log_f_ch('num_sample: ', str(self.num_sample)))
        logger.info(log_f_ch('n_class: ', str(self.label_n_class)))
        logger.info(log_f_ch('seq_len: ', str(self.seq_len)))
        logger.info(log_f_ch('n_channel: ', str(self.n_channel)))

    def __getitem__(self, index):
        data = load_mat(os.path.join(self.data_path,
                                     f'{self.data_list.iloc[index]["file"]}.h5'))
        if self.is_test:
            pd.DataFrame([[index, f'{self.data_list.iloc[index]["file"]}']]).to_csv(self.save_path, mode='a',index=False, header=False)
            return {
                'data': torch.from_numpy(data['amp']).float(),
                # 'freq_data': self.freq_data[index],
                'label': torch.from_numpy(data['label']).long()-1,
                'index': torch.tensor(index)
            }
        else:
            return {
                'data': torch.from_numpy(data['amp']).float(),
                'label': torch.from_numpy(data['label']).long()-1,
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
    datasource_path = os.path.join("D:\study\dataset\wifi-partition-data-abs\dataset")
    # datasource_path = os.path.join("D:\study\dataset\wifi-partition-data-abs\dataset")

    train_data, test_data = load_wifi_Vio_data(WiFiVioDatasetConfig(datasource_path))
    train_dataset = WiFiVioDataset(train_data)
    test_dataset = WiFiVioDataset(test_data)

    logger.info(len(train_dataset))
    logger.info(len(test_dataset))
    logger.info(train_dataset.get_n_classes(), test_dataset.get_n_classes())
    logger.info(train_dataset.get_n_channels(), test_dataset.get_n_channels())
    logger.info(train_dataset.get_seq_lens(), test_dataset.get_seq_lens())

    logger.info(len(train_dataset))
    from torch.utils.data.dataloader import DataLoader
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                     drop_last=True)
    for i, data in enumerate(loader):
        logger.info(str(i) + str(data['data'].shape) + str(data['label']))


