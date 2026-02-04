![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4550e42786144905bfab8b80ac064728.png)
配网缺陷检测图像数据集，螺栓销钉缺失
1.配网销钉缺失检测图像数据集（1200多张，voc，销钉缺失）
2.配网缺陷检测图像数据集（3000多张，voc，销钉缺失与绑扎不规范缺陷![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f0e3ecf95bed4e88aa98f51596976522.png)
）![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed968102cbdf4b898f60fa019bd464e0.png)
针对配网缺陷检测图像数据集，特别是螺栓销钉缺失和绑扎不规范缺陷的检测，我们可以使用YOLOv8模型进行训练和评估。你提到的数据集包含两种类型的图像数据：

1. **配网销钉缺失检测图像数据集**：大约1200张图像，VOC格式标注。
2. **配网缺陷检测图像数据集**：大约3000张图像，VOC格式标注，包含销钉缺失和绑扎不规范缺陷。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/454e3d97e1a1412ba7d3e5aa7de4325d.png)

### 1. 环境准备

首先，确保你已经安装了必要的库和工具。你可以使用以下命令安装所需的库：

```bash
pip install torch torchvision
pip install numpy
pip install pandas
pip install matplotlib
pip install opencv-python
pip install pyyaml
pip install ultralytics
```

### 2. 数据集准备

假设你的数据集目录结构如下：

```
power_grid_defects_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── power_grid_defects.yaml
```

每个图像文件和对应的标签文件都以相同的文件名命名，例如 `0001.jpg` 和 `0001.xml`（对于VOC格式）。

### 3. VOC到YOLO格式转换

由于YOLOv8模型直接支持YOLO格式的标注，我们需要将VOC格式的标注转换为YOLO格式。可以使用以下Python脚本来完成这个转换：

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(voc_dir, yolo_dir, class_names):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for filename in os.listdir(voc_dir):
        if not filename.endswith('.xml'):
            continue

        xml_path = os.path.join(voc_dir, filename)
        yolo_path = os.path.join(yolo_dir, filename.replace('.xml', '.txt'))

        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        with open(yolo_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue

                class_id = class_names.index(class_name)
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # Convert to YOLO format (center_x, center_y, width, height) and normalize
                center_x = (xmin + xmax) / 2.0 / image_width
                center_y = (ymin + ymax) / 2.0 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

# 定义类别名称
class_names = ['Missing Pin', 'Improper Binding']

# 转换训练集、验证集和测试集
convert_voc_to_yolo('power_grid_defects_dataset/labels/train', 'power_grid_defects_dataset/labels_yolo/train', class_names)
convert_voc_to_yolo('power_grid_defects_dataset/labels/val', 'power_grid_defects_dataset/labels_yolo/val', class_names)
convert_voc_to_yolo('power_grid_defects_dataset/labels/test', 'power_grid_defects_dataset/labels_yolo/test', class_names)
```

### 4. 创建数据集配置文件

创建一个 `power_grid_defects.yaml` 文件，内容如下：

```yaml
train: ../power_grid_defects_dataset/images/train
val: ../power_grid_defects_dataset/images/val
test: ../power_grid_defects_dataset/images/test

nc: 2
names: ['Missing Pin', 'Improper Binding']
```

### 5. 安装YOLOv8

克隆YOLOv8仓库并安装依赖项：

```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

### 6. 训练模型

使用YOLOv8的训练脚本进行训练。确保你已经在 `power_grid_defects.yaml` 中指定了正确的路径。

```bash
yolo task=detect mode=train model=yolov8n.yaml data=power_grid_defects.yaml epochs=100 imgsz=640 batch=16
```

- `epochs=100`：设置训练轮数。
- `imgsz=640`：设置输入图像大小。
- `batch=16`：设置批量大小。根据你的GPU内存大小调整这个值。

### 7. 评估模型

训练完成后，可以使用YOLOv8的评估脚本来评估模型在验证集上的性能。

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=power_grid_defects.yaml
```

### 8. 测试模型

为了评估模型在测试集上的性能，可以使用以下命令：

```bash
yolo task=detect mode=test model=runs/detect/train/weights/best.pt data=power_grid_defects.yaml
```

### 9. 可视化预测结果

使用以下Python代码来可视化模型的预测结果。

```python
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/detect/train/weights/best.pt')

# 读取图像
image_path = 'power_grid_defects_dataset/images/test/0001.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行预测
results = model(image)

# 绘制预测结果
results.print()
results.show()
```

### 10. 模型优化

为了进一步优化模型，可以尝试以下方法：

- **调整超参数**：使用不同的学习率、批量大小、权重衰减等。
- **使用预训练模型**：使用预训练的YOLOv8模型作为初始化权重。
- **增加数据量**：通过数据增强或收集更多数据来增加训练集的多样性。
- **模型融合**：使用多个模型进行集成学习，提高预测的准确性。
- **更复杂的网络结构**：尝试使用更大的YOLOv8模型，如 `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`。
- **数据增强**：使用数据增强技术，如旋转、缩放、翻转等，以增加模型的鲁棒性。
- **类别平衡**：如果某些类别的样本数量不平衡，可以使用类别平衡技术，如过采样或欠采样。

### 11. 数据增强

使用 `albumentations` 库进行数据增强，以提高模型的泛化能力。

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(640, 640),  # 根据需要调整尺寸
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])
```

### 12. 自定义数据加载器

创建自定义的数据加载器来读取图像和标签。

```python
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PowerGridDefectDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed['image']
            labels = transformed['bboxes']

        return image, labels
```

### 13. 训练代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 超参数
batch_size = 16
num_epochs = 50
learning_rate = 0.001

# 数据加载器
train_dataset = PowerGridDefectDataset(train_image_dir, train_label_dir, transform=transform)
val_dataset = PowerGridDefectDataset(val_image_dir, val_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8n.yaml')
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard
writer = SummaryWriter('runs/power_grid_defects_detection')

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar('Training Loss', avg_train_loss, epoch)

    # 验证
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'yolov8_power_grid_defects.pth')
```

### 14. 总结

通过以上步骤，你可以成功地使用YOLOv8模型对配网缺陷检测的数据集进行训练、评估和可视化。如果你有任何问题或需要进一步的帮助，请随时告诉我。希望这些信息对你有帮助！

### 15. 额外建议

- **多尺度训练**：由于数据集中的图像可能具有不同的分辨率和视角，可以考虑使用多尺度训练来提高模型的鲁棒性。
- **迁移学习**：如果数据集中的某些类别与现有公开数据集中的类别相似，可以考虑使用这些公开数据集进行预训练，然后再在你的数据集上进行微调。
- **模型压缩**：如果最终模型需要部署到资源受限的设备上，可以考虑使用模型压缩技术，如剪枝、量化等，来减小模型的大小和计算需求。

