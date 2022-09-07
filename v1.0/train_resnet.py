import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from model_resnet import mymodel_resnet

if __name__ == '__main__':
    train_datas = datasets.mydatasets("./dataset/train")
    test_data = datasets.mydatasets("./dataset/test")
    train_dataloader = DataLoader(train_datas, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    m = mymodel_resnet().cuda()

    loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    total_step = 0

for i in range(1):
    print("i={}".format(i))
    for ii, (imgs, targets) in enumerate(train_dataloader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = m(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ii % 10 == 0:
            total_step += 1
            print("训练{}次,loss:{}".format(total_step*10, loss.item()))
    # if i % 100 == 99:
    #     torch.save(m.state_dict(), "model_resnet_{}.pth".format(i+1))

torch.save(m.state_dict(), 'model_resnet.pth')
