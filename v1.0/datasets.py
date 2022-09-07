import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import onehot


class mydatasets(Dataset):

    def __init__(self, root_dir) -> None:
        super(mydatasets, self).__init__()
        self.list_image_path = [os.path.join(
            root_dir, image_name) for image_name in os.listdir(root_dir)]
        self.transforms = transforms.Compose([
            transforms.Resize((50, 100)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])

    def __getitem__(self, index):
        image_path = self.list_image_path[index]
        img_ = Image.open(image_path)
        image_name = image_path.split("\\")[-1]
        # image_name = image_path.split("/")[-1] # use in linux
        img_tensor = self.transforms(img_)
        img_lable = onehot.text2vec(image_name.split(".")[0])
        img_lable = img_lable.view(1, -1)[0]
        return img_tensor, img_lable

    def __len__(self):
        return self.list_image_path.__len__()
