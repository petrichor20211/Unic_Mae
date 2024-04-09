import torchvision.transforms as T
from torchvision import models
from PIL import Image
import torch

# 加载预训练的语义分割模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# 加载并对输入图片进行预处理
input_image = Image.open(r'D:\Torch\mae-main\data\DSCF0469.JPG')

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # 添加一个 batch 维度

# 将输入数据传入模型并获取输出
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# 将输出的预测结果转换为图像
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")
segmentation_mask = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
segmentation_mask.putpalette(colors)

# 显示原图和语义分割结果
input_image.show()
segmentation_mask.show()