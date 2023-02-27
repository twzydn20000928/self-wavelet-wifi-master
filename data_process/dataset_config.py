import os


class DatasetConfig(object):
    def __init__(self, dataset_name: str, datasource_path: os.path, save_path: os.path):
        self.dataset_name = dataset_name
        self.datasource_path = datasource_path
        self.save_path = save_path
