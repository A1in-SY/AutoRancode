# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import onehot


# class mymodel_trocr():
#     def __init__(self) -> None:
#         super(mymodel_trocr, self).__init__()
#         self.model = VisionEncoderDecoderModel.from_pretrained(
#             r'./trocr-base-printed')
#         # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(
#         #     7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # self.model.fc = nn.Linear(in_features=2048, out_features=onehot.rancode_size *
#         #                           onehot.rancode_array.__len__())

#     def forward(self, x):
#         x = self.model(x)
#         return x


# if __name__ == '__main__':
#     data = torch.randn(1, 1, 50, 100)
#     model = mymodel_trocr()
#     # x = model(data)
#     print(model)

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database (actually this model is meant to be used on printed text)
image = Image.open('./dataset/test/2ctm.png').convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
