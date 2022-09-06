from PIL import Image
from torch.utils.data import DataLoader
import onehot
import torch
import datasets
from torchvision import transforms
import net
from model_resnet import mymodel_resnet


use_model = './model_resnet_1000.pth'


def pred_imgs():
    # m = torch.load("model.pth").cuda()
    # m = torch.load(use_model).cuda()
    m = mymodel_resnet()
    m.load_state_dict(torch.load(use_model))
    print('Model loading complete.')
    m.eval()
    test_data = datasets.mydatasets("./dataset/test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    correct = 0
    for i, (imgs, lables) in enumerate(test_dataloader):
        # imgs = imgs.cuda() # when use .state_dict() to save the model, don't use .cuda()
        # lables = lables.cuda()
        lables = lables.view(-1, onehot.rancode_array.__len__())
        lables_text = onehot.vec2text(lables)
        predict_outputs = m(imgs)
        # print(predict_outputs.shape)
        predict_outputs = predict_outputs.view(-1,
                                               onehot.rancode_array.__len__())
        predict_labels = onehot.vec2text(predict_outputs)
        if predict_labels.lower() == lables_text.lower():
            correct += 1
            print("预测正确:正确值:{},预测值:{}".format(lables_text, predict_labels))
        else:
            print("预测失败:正确值:{},预测值:{}".format(lables_text, predict_labels))
        # input()
    print("正确率{}".format(correct / test_length * 100))


def pred_img(img_path, m):
    img = Image.open(img_path)
    tersor_img = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    img = tersor_img(img)
    # img = tersor_img(img).cuda()
    img = torch.reshape(img, (1, 1, 50, 100))
    predict_outputs = m(img)
    predict_outputs = predict_outputs.view(-1,
                                           onehot.rancode_array.__len__())
    predict_labels = onehot.vec2text(predict_outputs)
    return predict_labels


def net_pre(n):
    rate = 0
    m = mymodel_resnet()
    m.load_state_dict(torch.load(use_model))
    m.eval()
    print('Model loading complete.')
    for i in range(n):
        net.getimg()
        text = pred_img('./ocr.png', m)
        if net.check(text):
            rate += 1
            print('{} / {} 预测{}正确'.format(i+1, n, text))
        else:
            print('{} / {} 预测{}错误'.format(i+1, n, text))
        # input()
    print('准确率{}'.format(rate/n*100))


if __name__ == '__main__':
    # pred_imgs()
    # pred_img("./dataset/test/7u9o_1635053946.png")
    net_pre(100)
