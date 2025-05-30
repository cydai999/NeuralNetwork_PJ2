# README

### Introduction

本项目是神经网络与深度学习课程的PJ2， 共完成了两个任务：Task1利用CIFAR-10数据集训练神经网络， Task2旨在探究BatchNorm机制对训练过程的影响。下面将分别介绍如何在两个任务中训练和测试模型。

项目架构：

```
NeuralNetwork_PJ2
├── codes
│   ├── Task1
|   |    ├── models
|   |    ├── vis
|   |    ├── work_dir
|   |    ├── best_model.pth
|   |    ├── eval.py
|   |    ├── train.py
|   |    └── train_resnet20.py
│   └── Task2
|        └── VGG_Batchnorm
|   	      ├── loader
|   	      ├── log
|   	      ├── models
|   	      ├── vis
|   	      ├── plot_loss_landscape.py
|   	      ├── VGG_BN_train.py
|             └── VGG_train.py
└── data
    └── cifar-10-python.tar.gz
```



### Task1

#### 1.1 环境准备

需要用到的包为：

```
numpy
torch
torchvision
scikit-learn
matplotlib
tqdm
tensorboard
```

#### 1.2 数据集与模型

本项目所用数据集为CIFAR-10，可在 https://drive.google.com/file/d/1Aa3cNUeZRvHtqlX5C0dAhq2_fbjsyyWC/view?usp=drive_link 下载`cifar-10-python.tar.gz`文件， 下载后需放在`NeuralNetwork_PJ2/data`下（需新建目录`data`）。

模型文件的下载链接为 https://drive.google.com/file/d/1c9kHG7qxXmTzuGcXkbqHC7HvMizcIzW8/view?usp=drive_link ，下载后需放在`NeuralNetwork_PJ2/codes/Task1`下。

准备完毕后文件结构应基本同前述项目架构一致。

#### 1.3 训练与评估

##### 1.3.1 训练

首先转移到Task1目录下：

```
cd NeuralNetwork_PJ2/codes/Task1
```

想要从头开始训练ResNet-18模型，可以输入：

```
python train.py
```

注意：如果使用windows系统需要输入：

```
python train.py --num_workers 0
```

否则可能引起进程堵塞。之后同样需要设置`num_workers`为0， 不再特意说明。

如果想要训练ResNet-20模型，可以输入：

```
python train_resnet20.py
```

训练完成后在`Task1/work_dir`目录下会出现以时间戳为名的文件夹， 其中应包含如下文件：训练日志`log.json`，模型架构文件`model_architecture.txt`，训练结果`result.txt`，模型参数`best_model.pth`以及一个tensorboard输出文件`events.out...`。

##### 1.3.2 评估

想要评估下载的`best_model.pth`， 可以在Task1目录下输入：

```
python eval.py
```

注意：如果想要评估模型文件在其他位置，需要输入模型路径：

```
python eval.py --model_path {YOUR/PATH/TO/MODEL}
```

#### 1.4 可视化

若想可视化训练过程，可以在终端使用tensorboard：

```
tensorboard --logdir={LOG/DIR}    # events.out...文件所在目录
```



### Task2

#### 2.1 环境准备

同1.1

#### 2.2 数据集与模型

数据集同1.2。由于本任务旨在探究BatchNorm对训练过程的影响，因此并未保存模型文件。

#### 2.3 训练

首先转移到Task2/VGG_Batchnorm目录下：

```
cd NeuralNetwork_PJ2/codes/Task2/VGG_Batchnorm
```

想要训练VGG模型，可以直接在终端输入：

```
python VGG_train.py
```

训练过程将记录在`vgg`目录下。

想要训练VGG_BatchNorm模型，则可以输入：

```
python VGG_BN_train.py
```

训练过程将记录在`vgg_bn`目录下。

#### 2.4 可视化

想要可视化min_curve和max_curve曲线，可以输入：

```
python plot_loss_landscape.py
```

