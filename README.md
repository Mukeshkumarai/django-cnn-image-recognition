## Django + CNN 图像识别

本项目包含下面可运行示例：

- DNN 基线（MNIST）
- CNN 训练与评估（MNIST）
- 单图预测脚本（加载训练好的权重）
- 可选：将模型以 Django 服务化（步骤指引）

> 说明：示例使用 PyTorch + TorchVision。Django 仅作为可选服务化演示，默认脚本以内训推理为主。

---

### 目录结构

```
Django_CNN图像识别项目/
  ├─ README.md
  ├─ requirements.txt
  └─ src/
      ├─ data/
      │   └─ mnist_dataset.py
      ├─ models/
      │   ├─ dnn.py
      │   └─ cnn.py
      ├─ train_dnn_mnist.py
      ├─ train_cnn_mnist.py
      └─ predict_mnist.py
```

---

### 环境准备

1) 创建虚拟环境（任选其一）
- Conda:
```
conda create -n mnist python=3.10
conda activate mnist
```
- venv:
```
python -m venv .venv
.\.venv\Scripts\activate
```

2) 安装依赖
```
pip install -r requirements.txt
```

3) 数据集
- TorchVision 会自动下载 MNIST 到本地缓存目录（首次运行需联网）。

---

### 训练与评估

- DNN 基线：
```
python src/train_dnn_mnist.py --epochs 5 --batch-size 128 --lr 1e-3 --save checkpoints/dnn_mnist.pth
```

- CNN：
```
python src/train_cnn_mnist.py --epochs 5 --batch-size 128 --lr 1e-3 --save checkpoints/cnn_mnist.pth
```

日志会输出每个 epoch 的 loss / accuracy，并在 `checkpoints/` 下保存权重文件。

---

### 单图预测

```
python src/predict_mnist.py --model cnn --ckpt checkpoints/cnn_mnist.pth --image path/to/digit.png
```

要求：输入为 28x28 灰度或任意尺寸单通道/三通道图片（脚本会自动转灰度并缩放到 28x28）。

---

### 可选：Django 服务化（指引）

1) 安装 Django：
```
pip install django
```
2) 创建项目与应用：
```
django-admin startproject mnist_web
cd mnist_web
python manage.py startapp infer
```
3) 在 `infer/views.py` 中加载 `cnn.py` 模型权重（如 `checkpoints/cnn_mnist.pth`），编写一个上传图片并返回预测的 API（POST /predict）。
4) 在 `mnist_web/urls.py` 中路由到 `infer.views.predict`。
5) 启动服务：
```
python manage.py runserver 0.0.0.0:8000
```

> 提示：服务端加载模型时请设置 `model.eval()` 并在推理处使用 `torch.inference_mode()`；将权重路径放在安全可读位置。

---

### 常见问题

- 下载数据慢：可配置国内镜像或预先下载 MNIST 到本地缓存。
- 精度较低：适当增大 `epochs`、使用更强数据增强、或采用更深层 CNN 结构与更规范的优化超参。
- Windows + CPU：训练会较慢，建议减少 batch size 或迁移到支持 CUDA 的环境。


