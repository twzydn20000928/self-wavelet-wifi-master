import os

from scripts.utils import *

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 0
    # segment_length_rates = [2, 1, 0.5]
    segment_length_rates = [2, 1, 0.5]
    # overlap_length_rates = [0.75, 0.5, 0.25, 0]
    overlap_length_rates = [0.75, 0.5, 0.25, 0]

    dataset_list = []
    # ARIL
    for segment_length_rate in segment_length_rates:
        for overlap_length_rate in overlap_length_rates:
            dataset_list.append("aril_none_%.2f_%.2f" % (segment_length_rate, overlap_length_rate))
    # WiFi_AR
    for i in range(5):
        for segment_length_rate in segment_length_rates:
            for overlap_length_rate in overlap_length_rates:
                dataset_list.append("wifi_ar_%d_%.2f_%.2f" % (i, segment_length_rate, overlap_length_rate))
    # HTHI
    for i in range(1):
        for segment_length_rate in segment_length_rates:
            for overlap_length_rate in overlap_length_rates:
                dataset_list.append("hthi_%d_none_%.2f_%.2f" % (i, segment_length_rate, overlap_length_rate))

    model_list = [
        ('resnet1d_18', 'resnet1d_span_cls_raw_time', 128),
        ('resnet1d_18', 'resnet1d_span_cls_raw_channel', 128),
        ('resnet1d_18', 'resnet1d_span_cls_freq_time', 128),
        ('resnet1d_18', 'resnet1d_span_cls_freq_channel', 128),

        ('resnet2d_18_2', 'resnet2d_span_cls_raw', 128),
        ('resnet2d_18_2', 'resnet2d_span_cls_freq', 128),

        ('vit_es_2', 'vit_span_cls_freq', 128),
        ('vit_es_16', 'vit_span_cls_raw', 128),

        ('lstm_1', 'lstm_span_cls_raw', 128),
        ('lstm_1', 'lstm_span_cls_freq', 128),

        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rtrc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rtfc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rcfc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_ftfc', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rcft', 128),
        ('ds_resnet1d_18', 'ds_resnet1d_span_cls_rtft', 128),

        ('ds_resnet2d_18_2', 'ds_resnet2d_span_cls', 16),

        ('ds_vit_es_16_2_early', 'ds_vit_span_cls', 128),
        ('ds_vit_es_16_2_late', 'ds_vit_span_cls', 128),
        ('ds_vit_es_16_2_fully', 'ds_vit_span_cls', 128),

        ('ds_lstm_1', 'ds_lstm_span_cls', 128),

        ('fs_resnet1d_18', 'fs_resnet1d_span_cls', 128),
    ]

    for dataset_name in dataset_list:
        for module in model_list:
            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]
            os.system(
                './script_run.sh %d %s %s %s %s %d' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size)
            )
