import os

from scripts.utils import *


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 0

    config = DatasetDefaultConfig()

    model_list = [

        # ('wavevit_wave2_0_b_32', 'vit_span_cls_raw', 128),
        # ('wavevit_wave2_10_b_32', 'vit_span_cls_raw', 128),
        # ('wavevit_wave2_12_b_32', 'vit_span_cls_raw', 128),

        ('wavevit_wavelh2_0_s_16', 'vit_span_cls_raw', 128),
        ('wavevit_wavelh2_4_s_16', 'vit_span_cls_raw', 128),
        ('wavevit_wavelh2_8_s_16', 'vit_span_cls_raw', 128),
        # ('wavevit_wavelh2_12_b_32', 'vit_span_cls_raw', 128),
    ]
    config.dataset_list.append(f'WiAR_0.8')
    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:
            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]


            eval_batch_size = 1
            num_epoch = 500

            opt_method = "adamw"
            lr_rate = 2e-4
            weight_decay = 1e-4
            lr_rate_adjust_epoch = 100
            lr_rate_adjust_factor = 0.2
            save_epoch = 501
            eval_epoch = 501
            patience = 50

            test_batch_size = batch_size
            train_batch_size = batch_size
            # print(
            #     './script_run.sh %d %s %s %s %s %d' %
            #     (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size)
            # )
            # os.chdir("D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi")
            os.chdir("D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi")

            os.system('conda activate test')
            os.system(
                f'python main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
                --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
                --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --num_epoch {num_epoch} \
                --opt_method {opt_method} --lr_rate {lr_rate} --weight_decay {weight_decay} \
                --lr_rate_adjust_epoch {lr_rate_adjust_epoch} --lr_rate_adjust_factor {lr_rate_adjust_factor}  \
                --save_epoch {save_epoch} --eval_epoch {eval_epoch} --patience {patience} --is_train true \
                > {dataset_name}-{backbone_name}-{strategy_name}-TRAIN.log'
            )
            # os.system(f'PYTHON main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
            #             --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
            #             --test_batch_size {test_batch_size} \
            #             > {dataset_name}-{backbone_name}-{strategy_name}-TEST.log')
