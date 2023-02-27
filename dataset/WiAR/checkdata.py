import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
from util import log_f_ch


# 数据集大小：	557
# 数据size：	    torch.Size([4, 2000, 90])
# batch_size:	4
# label:    	tensor([0, 0, 0, 0])

if __name__ == '__main__':

    import os
    path1 = os.path.join("D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi\dataset")
    path2 = os.path.join("WiAR\Data.pt")
    path3 = os.path.join(path1, path2)
    print(path3)

    dataset = torch.load(path3)

    loader = Data.DataLoader(
        dataset = dataset,
        batch_size= 4,
        shuffle=False,
        num_workers=1,
    )
    print(type(dataset))
    print(type(loader))
    amp, label = next(iter(loader))

    # plt.figure(figsize=(20, 25))
    # for i in range(amp.shape[0]):
    #     plt.subplot(amp.shape[0],1,i+1)
    #     plt.plot(amp[i,:,:])
    #
    # plt.show()

    print(log_f_ch('数据集大小：',str(len(dataset))))
    print(log_f_ch('数据size：',str(amp.shape)))
    print(log_f_ch('batch_size:', str(4)))
    print(log_f_ch('label: ', str(label)))
    # print(type(amp.data))
    # print(label)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [445, 112])

    print(type(amp.float()))
    print(label.long())
    # amp, label = next(iter(dataset))
    # amp = amp.transpose(1, 0)
    # print(amp.shape)
    # for i in range(90):
    #     plt.plot(amp[i, :])
    # plt.show()

    # print(dataset.size())



