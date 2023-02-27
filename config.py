import os


class BasicConfig(object):
    def __init__(self):
        # backbone/head/strategy
        self.backbone_name = 'ds_resnet1d_34'
        self.head_name = 'wifi_ar_span_cls'
        self.strategy_name = 'ds_resnet1d_span_cls_rtrc'

        # 数据集名及其相应数据源路径
        '''
        aril_(none|pha)_(segment_length_rate)_(overlap_length_rate)
        wifi_ar_(index)
        hthi_(index)
        '''
        self.dataset_name = 'wifi_ar_0'
        self.datasource_path = os.path.join('/home/wuxilei/data/wifi_har_empirical_study')

        self.check_point_path = os.path.join(
            '/home/wuxilei/data/wifi_har_empirical_study/checkpoint/%s/%s' % (self.dataset_name, self.strategy_name))
        self.gpu_device = "1"


class TrainConfig(BasicConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()
        # 训练超参
        self.train_batch_size = 16
        self.eval_batch_size = 4
        self.num_epoch = 10
        self.opt_method = 'adam'
        self.lr_rate = 1e-4
        self.lr_rate_adjust_epoch = 20
        self.lr_rate_adjust_factor = 0.5
        self.weight_decay = 1e-4
        self.save_epoch = 10
        self.eval_epoch = 10
        self.patience = 10


class TestConfig(BasicConfig):
    def __init__(self):
        super(TestConfig, self).__init__()
        # 测试超参
        self.test_batch_size = 16

        self.output_path = self.check_point_path
