import os
import csv
import torch
import torch.nn as nn
import warnings
from datetime import datetime
from ultralytics import YOLO

warnings.filterwarnings('ignore', category=DeprecationWarning)


# ================= 1. LFEM 模块 =================
class LFEM(nn.Module):
    def __init__(self, in_channels):
        super(LFEM, self).__init__()
        self.in_channels = in_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            fea = x
        else:
            fea = x[0]

        f1 = self.branch1(fea)
        f2 = self.branch2(fea)
        f3 = self.branch3(fea)
        f4 = self.branch4(fea)

        fea_cat = torch.cat([f1, f2, f3, f4], dim=1)

        if self.in_channels % 32 == 0:
            batch_size, channels, height, width = fea_cat.size()
            channels_per_group = channels // 32
            fea_cat = fea_cat.view(batch_size, 32, channels_per_group, height, width)
            fea_cat = fea_cat.permute(0, 2, 1, 3, 4).contiguous()
            fea_cat = fea_cat.view(batch_size, -1, height, width)

        new_fea = self.fusion(fea_cat)
        return new_fea + fea


# ================= 2. ELA 模块 =================
class ELA(nn.Module):
    def __init__(self, channels):
        super(ELA, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            fea = x
        else:
            fea = x[0]
        y = self.conv(fea)
        y = self.conv1x1(y)
        y = self.sigmoid(y)
        return fea * y


# ================= 3. 包装函数 =================
def get_out_channels(layer):
    if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
        return layer.cv2.conv.out_channels
    if hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
        return layer.cv1.conv.out_channels
    if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
        return layer.conv.out_channels
    if hasattr(layer, 'c2'):
        return layer.c2
    return None


def wrap_layer(model, idx, module_class):
    layers = model.model.model
    if idx >= len(layers):
        print(f"索引 {idx} 超出范围（共{len(layers)}层），跳过")
        return False
    old = layers[idx]
    out_ch = get_out_channels(old)
    if out_ch is None:
        print(f"无法获取第 {idx} 层 ({old.__class__.__name__}) 的输出通道，跳过")
        return False
    new = nn.Sequential(old, module_class(out_ch))
    layers[idx] = new
    print(f"包装层{idx}: {old.__class__.__name__} (out_ch={out_ch}) -> Sequential(原层, {module_class.__name__})")
    return True


def verify_modules(model):
    layers = model.model.model
    found_ela = found_lfem = False
    for i, layer in enumerate(layers):
        layer_str = str(layer)
        if 'ELA' in layer_str and 'LFEM' not in layer_str:
            found_ela = True
            print(f"  层{i}: 发现 ELA")
        if 'LFEM' in layer_str:
            found_lfem = True
            print(f"  层{i}: 发现 LFEM")
    print(f"\n  ELA  已插入: {'是' if found_ela else '否'}")
    print(f"  LFEM 已插入: {'是' if found_lfem else '否'}")
    if not (found_ela and found_lfem):
        print("  警告: 有模块未成功插入！")
        return False
    print("  全部模块已成功插入！")
    return True


def save_summary_to_csv(project_dir, class_names, metrics, extra_info=""):
    os.makedirs(project_dir, exist_ok=True)
    csv_path = os.path.join(project_dir, "results_summary.csv")
    try:
        precision = metrics.box.mp
        recall = metrics.box.mr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        map50 = metrics.box.map50
        map5095 = metrics.box.map
    except Exception:
        precision = recall = f1 = map50 = map5095 = 0.0

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["=== YOLO11s + P2 + LFEM + ELA 训练验证总结报告 ==="])
        writer.writerow(["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["实验配置", extra_info])
        writer.writerow([])
        writer.writerow(["整体核心指标"])
        writer.writerow(["指标名称", "数值"])
        writer.writerow(["Precision (精确率)", f"{precision:.4f}"])
        writer.writerow(["Recall (召回率)", f"{recall:.4f}"])
        writer.writerow(["F1-Score", f"{f1:.4f}"])
        writer.writerow(["mAP@0.5", f"{map50:.4f}"])
        writer.writerow(["mAP@0.5:0.95", f"{map5095:.4f}"])
        writer.writerow([])
        writer.writerow(["各类别详细性能"])
        writer.writerow(["类别名称", "精确率", "召回率", "F1分数", "mAP@0.5"])
        for i, name in enumerate(class_names):
            try:
                p = metrics.box.p[i] if i < len(metrics.box.p) else 0
                r = metrics.box.r[i] if i < len(metrics.box.r) else 0
                ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
                f1_i = 2 * p * r / (p + r) if (p + r) > 0 else 0
                writer.writerow([name, f"{p:.4f}", f"{r:.4f}", f"{f1_i:.4f}", f"{ap50:.4f}"])
            except:
                pass
    print(f"\n总结表格已保存至: {csv_path}")


# ================= 4. 主程序 =================
if __name__ == '__main__':
    # ========== 路径配置 ==========
    yaml_path = r"D:\yolov11\pythonProject\ultralytics\cfg\models\11\yolo11-P2.yaml"
    model_path = r"D:\yolov11\pythonProject\yolo11s.pt"
    data_yaml_path = r"D:\Dataset for Solar Photovoltaic Panels\SPHERE - LABELLED DATASET\YOLO\data.yaml"
    project_name = 'runs/V11train'

    # ========== ★★★ 只改这里做 lr/epoch 对比实验 ★★★ ==========
    LR0 = 0.0002
    EPOCHS = 200
    WARMUP_EPOCHS = 10
    BATCH_SIZE = 16
    WORKERS = 8

    ELA_IDX = 25
    LFEM_IDX = 28

    EXPERIMENT_NAME = f'p2_lfem_ela_lr{LR0}_ep{EPOCHS}'
    # ============================================================

    class_names = ['broken', 'clean', 'dirty']

    print("=" * 60)
    print("YOLO11s + P2 + LFEM + ELA")
    print(f"  lr0 = {LR0}, epochs = {EPOCHS}, warmup = {WARMUP_EPOCHS}")
    print(f"  ELA  -> 层{ELA_IDX}（P4）")
    print(f"  LFEM -> 层{LFEM_IDX}（P5）")
    print("=" * 60)

    for p, name in [(yaml_path, 'YAML'), (model_path, '预训练权重'), (data_yaml_path, '数据集')]:
        if not os.path.exists(p):
            print(f"❌ {name}不存在: {p}")
            exit(1)

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # ========== 第一步：用 P2 yaml 建结构（yaml里只有s，默认就是s）==========
    model = YOLO(yaml_path)
    print("✅ 已用 yolo11-p2.yaml 建结构（默认s）")

    # ========== 第二步：加载预训练权重 ==========
    model = model.load(model_path)
    print(f"✅ 已加载预训练权重: {model_path}")

    # ========== 第三步：打印结构 ==========
    print("\n📋 模型层索引（前35层）:")
    for i, layer in enumerate(model.model.model[:35]):
        print(f"{i:3d}: {layer.__class__.__name__}")

    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n📋 模型总参数量: {total_params/1e6:.2f}M")
    print(f"📋 检测头数量: {len(model.model.model[-1].cv2)} 个")

    if len(model.model.model[-1].cv2) != 4:
        print("❌ 检测头不是4个，P2没生效！终止！")
        exit(1)
    if total_params < 8e6:
        print("❌ 参数量<8M，可能是n版本，终止！")
        exit(1)

    # ========== 第四步：插入模块 ==========
    print("\n正在插入模块...")
    wrap_layer(model, ELA_IDX, ELA)
    wrap_layer(model, LFEM_IDX, LFEM)

    print("\n验证模块插入状态:")
    if not verify_modules(model):
        print("模块插入失败，终止训练！")
        exit(1)

    # ========== 第五步：训练 ==========
    print("\n🚀 开始训练...")
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=project_name,
        name=EXPERIMENT_NAME,
        patience=50,
        verbose=True,
        lr0=LR0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        workers=WORKERS,
        device=0,
        optimizer='auto',
        seed=42,
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    # ========== 验证 ==========
    print("\n" + "=" * 50)
    print("✅ 训练结束！正在验证...")
    print("=" * 50)

    best_model_path = os.path.join(project_name, EXPERIMENT_NAME, 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"❌ 未找到 best.pt: {best_model_path}")
        exit(1)

    best_model = YOLO(best_model_path)
    metrics = best_model.val(
        data=data_yaml_path,
        split='val',
        imgsz=640,
        batch=BATCH_SIZE,
        plots=True,
        verbose=True,
        save_json=True,
        augment=False,
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "*" * 60)
    print(f"🏆 YOLO11s + P2 + LFEM + ELA (lr={LR0}, ep={EPOCHS}) 评估成绩单 🏆")
    print("*" * 60)
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数 (F1-Score):  {f1:.4f}")
    print(f"  mAP@0.5:            {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:       {metrics.box.map:.4f}")

    print(f"\n{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'mAP@0.5':<10}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        p = metrics.box.p[i] if i < len(metrics.box.p) else 0
        r = metrics.box.r[i] if i < len(metrics.box.r) else 0
        ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        f1_i = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{name:<12} {p:<10.4f} {r:<10.4f} {f1_i:<10.4f} {ap50:<10.4f}")

    result_dir = os.path.join(project_name, EXPERIMENT_NAME)
    save_summary_to_csv(result_dir, class_names, metrics, extra_info=f"lr={LR0}, ep={EPOCHS}")

    print("\n" + "*" * 60)
    print(f"📁 结果保存至: {result_dir}")
    print("*" * 60)