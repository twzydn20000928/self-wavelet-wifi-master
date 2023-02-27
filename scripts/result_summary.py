from scripts.utils import *

if __name__ == '__main__':
    checkpoint_path = os.path.join('/home/wuxilei/data/wifi_har_empirical_study/checkpoint')
    config = DatasetDefaultConfig()

    generate_result_summary(checkpoint_path, config.dataset_list)
