## 概览

要从头开始实现缺陷检测的**Fine‑tuning**，主要流程包括：

1. **选择预训练框架**（如 Detectron2、MMDetection、YOLOv5/YOLOv8），
2. **准备数据集**（标注、转为 COCO/YOLO 格式），
3. **搭建环境**（安装依赖、克隆仓库），
4. **修改配置文件**（指定类别、冻结层数等），
5. **启动训练**（设置训练参数），
6. **超参数调优**（学习率、Momentum 等）与
7. **模型评估和部署**。

下面将逐步展开每个环节的具体操作和注意事项。

---

## 1. 选择合适的预训练框架

### Detectron2

* Facebook 出品，基于 PyTorch，简洁易用。可直接从 Model Zoo 拉取 Faster R‑CNN、Mask R‑CNN 等检测模型，并对自定义数据微调（fine‑tune）【([Medium][1])】【([Roboflow Blog][2])】。

### MMDetection

* OpenMMLab 生态下的 Detection 工具箱，同样提供丰富的预训练模型和配置继承机制。通过继承 `_base_/models/...` 和 `_base_/datasets/...` 等配置，即可快速适配新数据集【([MMDetection][3])】【([MMDetection][4])】。

### YOLO 系列

* **YOLOv5**：Ultralytics 提供详细教程，数据准备、配置和训练只需数行代码，可快速上手【([Ultralytics Docs][5])】。
* **YOLOv8**：最新研究表明，通过分层解冻（unfreeze）微调可以在新任务上大幅提升 mAP，同时保持 COCO 基准性能不变【([arXiv][6])】。

---

## 2. 数据集准备

1. **收集与标注**：使用 LabelMe、Makesense.ai 或 Roboflow 进行物体边界框（bounding box）标注。
2. **格式转换**：

   * COCO 格式：常见于 Detectron2/MMDetection，需生成 `annotations.json`，键包括 `images`, `annotations`, `categories`【([MMDetection][3])】。
   * YOLO 格式：每张图片对应一个 `.txt` 文件，行格式 `<class_id> <x_center> <y_center> <width> <height>`（归一化坐标）【([DigitalOcean][7])】。
3. **数据划分**：一般按 8:1:1 划分训练/验证/测试集，确保类别分布均匀。

---

## 3. 环境搭建与依赖安装

以 Detectron2 为例：

```bash
# 安装 Detectron2（需匹配 PyTorch 版本）
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

以 MMDetection 为例：

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

以 YOLOv5 为例：

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

以上步骤均在 Python≥3.8、PyTorch≥1.8 环境下进行【([Ultralytics Docs][5], [GitHub][8])】。

---

## 4. 修改配置文件

### Detectron2

* 在示例配置 `configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml` 上：

  1. 修改 `DATASETS.TRAIN`、`DATASETS.TEST` 指向自定义数据集名称；
  2. 调整 `SOLVER.BASE_LR`, `SOLVER.MAX_ITER` 等；
  3. 可通过 `cfg.MODEL.BACKBONE.FREEZE_AT` 决定冻结层级。

### MMDetection

* 编写新配置继承：

  ```yaml
  _base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
  ]
  model = dict(
    roi_head=dict(
      bbox_head=dict(num_classes=你的类别数)
    )
  )
  data = dict(
    train=dict(ann_file='path/to/train.json', img_prefix='train/'),
    val=dict(ann_file='path/to/val.json', img_prefix='val/')
  )
  ```
* 继承避免重复配置，并保证与 Model Zoo 兼容【([MMDetection][3])】。

### YOLOv5/YOLOv8

* 编辑 `data.yaml`：

  ```yaml
  train: path/to/images/train
  val:   path/to/images/val
  nc:    类别数
  names: ['defect1','defect2',...]
  ```
* 选择预训练权重（如 `yolov5s.pt`），并在训练命令中指定冻结层数：

  ```bash
  python train.py --img 640 --batch 16 --epochs 50 \
    --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt \
    --freeze 10
  ```

  其中 `--freeze` 控制冻结前 N 层，参考最新研究可根据任务需求调整深度【([arXiv][6])】。

---

## 5. 启动训练

* **Detectron2**：

  ```bash
  python tools/train_net.py \
    --config-file configs/YourConfig.yaml \
    --num-gpus 1
  ```
* **MMDetection**：

  ```bash
  ./tools/dist_train.sh \
    configs/your/finetune_config.py 1
  ```
* **YOLOv5**：

  ```bash
  python train.py \
    --data data.yaml --weights yolov5s.pt \
    --epochs 50 --batch-size 16
  ```

训练过程中可实时输出 mAP、loss 曲线，并在验证集上监控性能。

---

## 6. 超参数与调优

* **学习率 (LR)**：一般初始设为 `1e-3`\~`1e-4`，可使用 Warmup 与 LR Scheduler（Step、Cosine）进行动态调整【([arXiv][9])】。
* **Momentum & Weight Decay**：Momentum 建议 0.9 左右，Weight Decay 可尝试 `1e-4`\~`1e-5`，视任务和数据量微调【([arXiv][9])】。
* **冻结策略**：如 YOLOv8 研究所示，逐步解冻（freeze points）可提升细粒度检测性能，同时避免遗忘原模型能力【([arXiv][6])】。

---

## 7. 模型评估与部署

1. **评估**：使用 COCO API 计算 AP、mAP、Recall 等指标；
2. **导出权重**：Detectron2 可导出 `.pth`，MMDetection 同理；YOLOv5 支持导出 ONNX/ TorchScript；
3. **部署**：

   * 云端服务（Flask/GPU）；
   * 边缘端（TensorRT、OpenVINO）；
   * 集成到 ROS/工业流水线。

---

通过上述步骤，你即可从零开始完成基于预训练模型的缺陷检测 Fine‑tuning。根据项目需求，灵活选择框架和超参数，并在训练中不断迭代优化。祝你早日实现高精度、实时的缺陷检测系统！

[1]: https://medium.com/the-owl/guide-to-fine-tuning-a-pre-trained-model-for-object-detection-tasks-with-faster-rcnn-using-e2e7e190105b?utm_source=chatgpt.com "Guide to fine-tuning a Pre-trained model for Object Detection tasks ..."
[2]: https://blog.roboflow.com/how-to-train-detectron2/?utm_source=chatgpt.com "How to Train Detectron2 on Custom Object Detection Data"
[3]: https://mmdetection.readthedocs.io/en/v2.9.0/tutorials/finetune.html?utm_source=chatgpt.com "Tutorial 7: Finetuning Models - MMDetection's documentation!"
[4]: https://mmdetection.readthedocs.io/en/latest/user_guides/train.html?utm_source=chatgpt.com "Train predefined models on standard datasets"
[5]: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/?utm_source=chatgpt.com "Train YOLOv5 on Custom Data - Ultralytics YOLO Docs"
[6]: https://arxiv.org/abs/2505.01016?utm_source=chatgpt.com "Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance"
[7]: https://www.digitalocean.com/community/tutorials/train-yolov5-custom-data?utm_source=chatgpt.com "How to Train YOLO v5 on a Custom Dataset | DigitalOcean"
[8]: https://github.com/open-mmlab/mmdetection?utm_source=chatgpt.com "open-mmlab/mmdetection: OpenMMLab Detection Toolbox ... - GitHub"
[9]: https://arxiv.org/abs/2002.11770?utm_source=chatgpt.com "Rethinking the Hyperparameters for Fine-tuning"
