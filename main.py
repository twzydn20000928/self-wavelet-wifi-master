import os
import argparse
import logging
import torch

from config import BasicConfig, TrainConfig, TestConfig
from training import train
from testing import test

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def init_configs() -> BasicConfig:
    parser = argparse.ArgumentParser(description="WiFi Human activity recognition empirical study")

    parser.add_argument('--is_train', dest="is_train", required=False, type=bool, default=False,
                        help="是否训练")

    """模型/预测头/策略"""
    parser.add_argument('--backbone_name', dest="backbone_name", required=True, type=str,
                        help="骨干模型名称，内含相应的参数")
    parser.add_argument('--head_name', dest="head_name", required=True, type=str,
                        help="预测头名称")
    parser.add_argument('--strategy_name', dest="strategy_name", required=True, type=str,
                        help="模型训练方法的策略名，内含相应的参数")


    """数据集名及其相应数据源路径"""
    parser.add_argument('--dataset_name', dest="dataset_name", required=True, type=str,
                        help="数据集名，内含相应参数：随机索引及是否使用相位等")
    parser.add_argument('--datasource_path', dest="datasource_path", required=False, type=str,
                        # help="数据集源路径", default='D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi\dataset')
                        help="数据集源路径", default='D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi\dataset')
    parser.add_argument('--check_point_path', dest="check_point_path", required=False, type=str,
                        help="模型以及预测结果保存路径", default='/home/lanbo/wifi_wavelet/result/checkpoint/')
    parser.add_argument('--tab', dest='tab', required=False, type=str,
                        help="输出checkpoint的标记，避免覆盖", default='')

    parser.add_argument('--gpu_device', dest="gpu_device", required=True, type=str,
                        help="选择GPU设备编号，不适用GPU则-1")


    """训练超参"""
    parser.add_argument('--train_batch_size', dest="train_batch_size", required=False, type=int, default=64,
                        help="训练使用batch_size")
    parser.add_argument('--eval_batch_size', dest="eval_batch_size", required=False, type=int, default=64,
                        help="验证使用batch_size")
    parser.add_argument('--num_epoch', dest="num_epoch", required=False, type=int, default=200,
                        help="训练epoch")
    parser.add_argument('--opt_method', dest="opt_method", required=False, type=str, default="adam",
                        help="训练模型使用优化器")
    parser.add_argument('--lr_rate', dest="lr_rate", required=False, type=float, default=1e-4,
                        help="训练学习率")
    parser.add_argument('--lr_rate_adjust_epoch', dest="lr_rate_adjust_epoch", required=False, type=int, default=20,
                        help="每训练一定epoch后根据调整因子调整学习率")
    parser.add_argument('--lr_rate_adjust_factor', dest="lr_rate_adjust_factor", required=False, type=float,
                        default=0.5, help="每训练一定epoch后乘以学习率")
    parser.add_argument('--weight_decay', dest="weight_decay", required=False, type=float, default=1e-4,
                        help="训练正则化系数")
    parser.add_argument('--save_epoch', dest="save_epoch", required=False, type=int, default=50,
                        help="训练中途每隔一定epoch数后对模型进行保存")
    parser.add_argument('--eval_epoch', dest="eval_epoch", required=False, type=int, default=20,
                        help="训练中途每隔一定epoch数后使用模型在验证集上验证")
    parser.add_argument('--patience', dest="patience", required=False, type=int, default=10,
                        help="Early Stop机制，超过一定轮数eval loss未下降则停止训练")

    """测试相关参数"""
    parser.add_argument('--test_batch_size', dest="test_batch_size", required=False, type=int, default=64,
                        help="测试使用batch_size")

    # """分布式训练参数"""
    # parser.add_argument('--is_dist', dest="is_dist", required=False, type=bool, default=False)
    # parser.add_argument('--world_size', dest="world_size", required=False, type=int, default=1,
    #                     help='number of distributed processes')
    # parser.add_argument('--local_rank', dest="local_rank",required=False, default=-1, type=int)
    # # parser.add_argument('--dist_on_itp', dest="dist_on_itp", required=False, action='store_true')
    # parser.add_argument('--dist_url', dest="dist_url", required=False, default='env://',
    #                     help='url used to set up distributed training')

    args = parser.parse_args()

    configs = None
    if args.is_train:
        configs = TrainConfig()
    else:
        configs = TestConfig()

    # basic
    configs.backbone_name = args.backbone_name
    configs.head_name = args.head_name
    configs.strategy_name = args.strategy_name

    configs.dataset_name = args.dataset_name
    if configs.dataset_name.startswith('aril'):
        configs.datasource_path = os.path.join(args.datasource_path, 'aril')
    elif configs.dataset_name.startswith('WiAR'):
        configs.datasource_path = os.path.join(args.datasource_path, 'WiAR')
    elif configs.dataset_name.startswith('hthi'):
        configs.datasource_path = os.path.join(args.datasource_path, 'hthi')
    elif configs.dataset_name.startswith('WiVio'):
        configs.datasource_path = os.path.join(args.datasource_path)
    elif configs.dataset_name.startswith('AUGWiVio'):
        configs.datasource_path = os.path.join(args.datasource_path)

    configs.check_point_path = os.path.join(
        args.check_point_path, '%s' % configs.dataset_name, '%s' % configs.strategy_name
    )

    if args.tab != '':
        configs.check_point_path = os.path.join(configs.check_point_path, args.tab, '%s' % configs.backbone_name)
        
    if not os.path.exists(configs.check_point_path):
        os.makedirs(configs.check_point_path)

    configs.gpu_device = args.gpu_device

    # train
    configs.train_batch_size = args.train_batch_size
    configs.eval_batch_size = args.eval_batch_size
    configs.num_epoch = args.num_epoch
    configs.opt_method = args.opt_method
    configs.lr_rate = args.lr_rate
    configs.lr_rate_adjust_epoch = args.lr_rate_adjust_epoch
    configs.lr_rate_adjust_factor = args.lr_rate_adjust_factor
    configs.weight_decay = args.weight_decay
    configs.save_epoch = args.save_epoch
    configs.eval_epoch = args.eval_epoch
    configs.patience = args.patience

    # test
    configs.test_batch_size = args.test_batch_size
    configs.output_path = configs.check_point_path

    # if args.is_dist:
    #     configs.world_size = args.world_size
    #     configs.local_rank = args.local_rank
    #     configs.dist_url = args.dist_url

    return configs


if __name__ == '__main__':
    config = init_configs()
    if isinstance(config, TrainConfig):
        train(config)
    elif isinstance(config, TestConfig):
        test(config)
