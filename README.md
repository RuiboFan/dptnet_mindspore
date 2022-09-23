# 目录

<!-- TOC -->

- [目录](#目录)
- [DPTNet介绍](#DPTNet介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [数据预处理过程](#数据预处理过程)
        - [数据预处理](#数据预处理)
    - [训练过程](#训练过程)
        - [训练](#训练)  
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# DPTNet介绍

直接上下文感知的端到端语音分离网络（DPTNet）由三个处理阶段组成, 编码器、分离和解码器。首先，编码器模块用于将混合波形的短段转换为它们在中间特征空间中的对应表示。然后，使用transformer提取语音的中间特征。然后，利用解码器模块对屏蔽编码器特征进行变换，重构源波形。
DPTNet被广泛的应用在语音分离等任务上，取得了显著的效果

[论文](https://arxiv.org/abs/2007.13975): Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation

# 模型架构

模型包括  
encoder：类似fft，提取语音特征。
decoder：类似ifft，得到语音波形
separation：提取语音中间特征

# 数据集

使用的数据集为: [librimix](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>)，LibriMix 是一个开源数据集，用于在嘈杂环境中进行源代码分离。
要生成 LibriMix，请参照开源项目：https://github.com/JorisCos/LibriMix

# 环境要求

- 硬件（ASCEND）
    - ASCEND处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 依赖
    - 见requirements.txt文件，使用方法如下:

```python
pip install -r requirements.txt
 ```

# 脚本说明

## 脚本及样例代码

```path
DPTNet
├─ requirements.txt                   # requirements
├─ README.md                          # descriptions
├── scripts
  ├─ run_distribute_train.sh          # launch ascend training(8 pcs)
  ├─ run_stranalone_train.sh          # launch ascend training(1 pcs)
  ├─ run_eval.sh                      # launch ascend eval
  ├─ run_infer_310.sh                 # launch infer 310
├─ train.py                           # train script
├─ evaluate.py                        # eval
├─ preprocess.py                      # preprocess json
├─ data_loader.py                     # postprocess data
├─ export.py                          # export mindir script
├─ network_define.py                  # define network
├─ model.py                           # dptnet
├─ loss.py                            # loss function
├─ lr_sch.py                          # dynamic learning rate
├─ transformer.py                     # transformer module
├─ preprocess_310.py                  # preprocess of 310
├─ postprocess.py                     # postprocess of 310

```

## 脚本参数

数据预处理、训练、评估的相关参数在`train.py`等文件

```text
数据预处理相关参数
in_dir                    预处理前加载原始数据集目录
out_dir                   预处理后的json文件的目录
sample_rate               采样率
train_name                预处理后的训练MindRecord文件的名称
test_name                 预处理后的测试MindRecord文件的名称  
```

```text
训练和模型相关参数
train_dir                  训练集
valid_dir                  测试集
sample_rate                采样率
segment                    截取语音的长度
enc_dim                    一维卷积的卷积核数量
feature_dim                编码器的通道数
hidden_dim                 隐藏层
layer                      Transformer层数
segment_size               切割的语音块大小
nspk                       讲话人数
win_len                    窗口大小
epochs                     训练轮数
lr                         学习率
l2                         权重衰减
save_folder                模型保存路径
continue_train             是否继续训练
step_per_epoch             每个epoch的step数
ckpt_path                  权重路径
device_target              硬件配置
modelArts                  是否云上训练
```

```text
评估相关参数
ckpt_path                  ckpt文件
cal_sdr                    是否计算SDR
data-dir                   测试集路径
batch_size                 测试集batch大小
```

```text
配置相关参数
device_traget              硬件，只支持ASCEND
device_id                  设备号
```

# 数据预处理过程

## 数据预处理

数据预处理运行示例:

```text
python preprocess.py
```

数据预处理过程很快，大约需要三分钟时间

# 训练过程

## 训练

- ### 单卡训练

运行示例:

```text
python train.py
```

或者可以运行脚本:

```bash
./scripts/run_standalone_train.sh [DEVICE_ID]
```

上述命令将在后台运行，可以通过train.log查看结果  
每个epoch将运行1.5小时左右

- ### 分布式训练

分布式训练脚本如下

```bash
./scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE]
```

# 评估过程

## 评估

运行示例:

```text
python eval.py
参数:
model_path                 ckpt文件
data-dir                   测试集路径
batch_size                 测试集batch大小
```

或者可以运行脚本:

```bash
./scripts/run_eval.sh [DEVICE_ID]
```

上述命令在后台运行，可以通过eval.log查看结果

# 导出mindir模型

## 导出

```bash
python export.py
```

# 推理过程

## 推理

### 用法

```bash
./scripts/run_infer_310.sh [MINDIR_PATH] [TEST_PATH] [NEED_PREPROCESS]
```

### 结果

```text
Average SISNR improvement: 11.49
```

# 模型描述

## 性能

### 训练性能

| 参数                 |    DPTNet                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | Ascend910             |
| 上传日期              | 2022-8-26                                    |
| MindSpore版本           | 1.6.1                                                          |
| 数据集                    | Librimix                                                 |
| 训练参数       | 8p, epoch = 100, batch_size = 2   |
| 优化器                  | Adam                                                           |
| 损失函数              | SI-SNR                                |
| 输出                    | SI-SNR(11.54)                                                    |
| 损失值                       | -15.74                                                       |
| 运行速度                      | 8p 4882.254 ms/step                                   |
| 训练总时间       | 8p:约200h;                                  |                                           |

# 随机情况说明

随机性主要来自下面两点:

- 参数初始化
- 轮换数据集

# ModelZoo主页

 [ModelZoo主页](https://gitee.com/mindspore/models).
