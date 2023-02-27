import os
import torch
import logging

from torch.utils.data.dataloader import DataLoader

from pipeline import Tester
from config import TestConfig

import init_util

logger = logging.getLogger(__name__)


def test(config: TestConfig):
    if config.gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device

    if not os.path.exists(os.path.join(config.output_path, 'Test_dataset')):
        os.makedirs(os.path.join(config.output_path, 'Test_dataset'))
    if not os.path.exists(os.path.join(config.output_path, 'Train_dataset')):
        os.makedirs(os.path.join(config.output_path, 'Train_dataset'))

    train_dataset, eval_dataset = init_util.init_dataset(config.dataset_name, config.datasource_path, config.check_point_path, is_test=True)

    strategy = init_util.init_strategy(config.backbone_name,
                                       config.head_name,
                                       config.strategy_name,
                                       train_dataset.get_n_channels(),
                                       train_dataset.get_seq_lens())

    strategy.load_state_dict(torch.load(os.path.join(config.check_point_path, "%s-%s-final" % (
        config.backbone_name, config.head_name,
    ))))

    print('Test dataset'.center(100, '='))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(eval_dataset, batch_size=config.test_batch_size, shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=os.path.join(config.output_path, 'Test_dataset'),
        use_gpu=False if config.gpu_device is None else True
    )

    tester.testing()

    print('Train dataset'.center(100, '='))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=os.path.join(config.output_path, 'Train_dataset'),
        use_gpu=False if config.gpu_device is None else True
    )

    tester.testing()
    # 90 1000