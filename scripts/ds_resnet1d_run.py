import os

from scripts.utils import *

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 1

    config = DatasetDefaultConfig()

    model_list = [
        ('ds_resnet1d_101', 'ds_resnet1d_span_cls_rtfc', 128),
        ('ds_resnet1d_101', 'ds_resnet1d_span_cls_rcfc', 128),
        ('ds_resnet1d_101', 'ds_resnet1d_span_cls_rcft', 128),
        ('ds_resnet1d_101', 'ds_resnet1d_span_cls_rtft', 128),

        ('ds_resnet1d_50', 'ds_resnet1d_span_cls_rtfc', 128),
        ('ds_resnet1d_50', 'ds_resnet1d_span_cls_rcfc', 128),
        ('ds_resnet1d_50', 'ds_resnet1d_span_cls_rcft', 128),
        ('ds_resnet1d_50', 'ds_resnet1d_span_cls_rtft', 128),

        ('ds_resnet1d_34', 'ds_resnet1d_span_cls_rtfc', 128),
        ('ds_resnet1d_34', 'ds_resnet1d_span_cls_rcfc', 128),
        ('ds_resnet1d_34', 'ds_resnet1d_span_cls_rcft', 128),
        ('ds_resnet1d_34', 'ds_resnet1d_span_cls_rtft', 128),

        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rtfc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rcfc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rcft', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rtft', 128),
    ]

    for dataset_name in config.dataset_list:
        for module in model_list:
            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]
            os.system(
                './script_run.sh %d %s %s %s %s %d' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size)
            )
