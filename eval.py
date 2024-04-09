import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import models_mae

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model
chkpt_dir = 'model/mae_visualize_vit_large.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
torch.manual_seed(2)
# run_one_image(img, model_mae)

# 加载训练好的模型
model = model_mae
model.eval()  # 预测模式

# 加载CIFAR-10测试集
transform = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(root='./data_eval', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# 创建保存输出图片的目录
output_dir = './output_images/'
os.makedirs(output_dir, exist_ok=True)

# 遍历测试集，并保存输出图片
with torch.no_grad():
    for i, (inputs, _) in enumerate(testloader):
        # 将输入图像传递给模型进行预测
        outputs = model(inputs)

        # 保存输出图片
        output_path = os.path.join(output_dir, f'image_{i}.png')
        save_image(outputs, output_path)

print("所有输出图片已保存在", output_dir)