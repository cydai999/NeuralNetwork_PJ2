# 神经网络·PJ2

## 姓名：代承谕

## 学号： 22407130165



### Task1: Train a Network on CIFAR-10

#### 1.1 Introduction of the dataset

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32×32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. Since the images in CIFAR-10 are low-resolution (32 × 32), this dataset can allow us to quickly try our models to see whether it works.



#### 1.2 Basic Model

The basic model used in this project is **ResNet-series** neural networks. Compared to VGG, researchers of ResNet introduced *deep residual learning* framework, which effectively avoids gradient vanishing and exploding gradient, enabling deeper networks. According to experiments, the deep residual nets are not only easy to optimize, but also can easily enjoy accuracy gains from greatly increased depth. On the ImageNet dataset, an ensemble of these residual nets achieves 3.57% error, which took the 1st place on the ILSVRC 2015 classification task. Below is the origin architecture of ResNet-18:

<img src="D:\学习\大三\下学期\神经网络\PJ2\resnet18.drawio.png" alt="resnet18.drawio" style="zoom: 67%;" />

<center style="font-size:16px"><b>Figure 1: The architecture of ResNet-18</b></center>



#### 1.3 Experiments

During the experiment, I have tried various strategies to optimize my network: First of all, to get the baseline, I trained the **original ResNet-18** on CIFAR-10 without any adjustment; Then I slightly changed the architecture of ResNet-18 (**Figure 3**) to make it more suitable for CIFAR-10 dataset; After that, I tried various **regularization methods** including data augmentation, dropout, weight decay and early stopping. I also experimented on different optimizers, loss functions, activations and training time. Below are the details of the experiments.



##### 1.3.1 Original ResNet-18

Training configs:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 30
```

Training resuls:

```
[Train]Total training time:69.64393043518066
[Train]Training epochs:30
[Test]Accuracy:69.5500
```

Training curve:

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\original.png" alt="original" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 2: Training curves of original ResNet-18</b></center>

##### 1.3.2 Adjusted ResNet-18

Considering that the image in CIFAR-10 only has the resolution of 32*32, it's a little bit unsuitable to use a 7\*7 kernel at the first layer, as well as down sampling the images using a max pooling layer. Therefore, I transformed the first layer to a convolution layer with a **3\*3 kernel**  with **stride 1**. Furthermore, I added a **batch-norm layer** after each convolution layer. The adjusted ResNet-18 is shown in **Figure 3**:

<img src="D:\学习\大三\下学期\神经网络\PJ2\resnet18(revised).drawio.png" alt="resnet18(revised).drawio" style="zoom:67%;" />

<center style="font-size:16px"><b>Figure 3: The architecture of adjusted ResNet-18</b></center>

Training configs:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 30
```

Training results:

```
[Train]Total training time:164.93128490447998
[Train]Training epochs:30
[Test]Accuracy:79.2800
```

Training curve:

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\adjusted.png" alt="adjusted" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 4: Training curves of adjusted ResNet-18</b></center>

Comparison with origin ResNet-18:

|                        | Accuracy    | Training time(s) |
| ---------------------- | ----------- | ---------------- |
| **Original ResNet-18** | 69.5500     | 69.64            |
| **Adjusted ResNet-18** | **79.2800** | 164.93           |

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\compare.png" alt="compare" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 5: Comparison of original ResNet-18 and adjusted ResNet-18</b></center>

Compared to original ResNet-18, adjusted ResNet-18 achieves better accuracy (about $10\%$) at the cost of higher training time, likely due to the lack of down sampling at the first layer and extra FLOPs from batch-norm layers.



##### 1.3.3 Regularization Methods

During the training process of adjusted ResNet-18, I found there exists extreme overfitting. That is to say, the accuracy of training set is much higher than that of valid set, as **Figure 6** demonstrates.

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\compare_train_val.png" alt="compare_train_val" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 6: Comparison of train accuracy and valid accuracy based on adjusted ResNet-18</b></center>

To address the overfitting phenomenon, I have tried various regularization methods as follow:

**(1)** **Data augmentation**

Data augmentation is a useful technique to artificially expand the size and diversity of a training dataset by creating modified versions of existing data samples. In this project, I conducted data augmentation with some functions from `torchvision.transforms`:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])
```

**(2) Dropout**

Dropout means randomly abandoned some neurons during the forward process  in order to avoid overfitting.  In this project, I only added a dropout layer before the fully connected layer and tried different dropout rate ranging from 0.05-0.5.

**(3) Early Stopping**

Early stopping is another convenient but useful strategy to address overfitting and shorten the training time. It requires stopping training as soon as the accuracy on valid set stops rising. In this project, I set the patience of stopping as 5, which means stopping training when the accuracy on valid set doesn't climb for 0.1% in 5 continuous epochs.

The effect of those regularization methods are shown below: ("DA" means data augmentation, "ES" means early stopping.)

| Methods                       | Accuracy    | Training time(s) |
| ----------------------------- | ----------- | ---------------- |
| **Adjusted**                  | 79.2800     | 164.93           |
| **Adjusted+DA**               | 87.8000     | 236.08           |
| **Adjusted+Dropout(0.05)**    | 81.0100     | 167.33           |
| **Adjusted+Dropout(0.1)**     | 83.2700     | 164.35           |
| **Adjusted+Dropout(0.5)**     | 79.8100     | 164.55           |
| **Adjusted+ES**               | 79.2000     | 142.52           |
| **Adjusted+DA+Dropout(0.05)** | 87.9300     | 235.97           |
| **Adjusted+DA+Dropout(0.1)**  | **88.2700** | 236.33           |
| **Adjusted+DA+Dropout(0.5)**  | 87.1800     | 233.78           |

**Analysis:** From the tabular above, we could figure out the real effect of those regularization methods: data augmentation method achieved significant improvement for about $8\%$, while dropout with 0.1 rate brings a less improvement for about $4\%$. When combined together, data augmentation and dropout contribute to a total of $9\%$ accuracy gain. For another thing, although early stopping seemed to have no effect for addressing overfitting, it saved the training time for about $13.3\%$ with little loss of accuracy.

Following experiments will inherit the best configs, i.e., **Adjusted+DA+Dropout(0.1)+ES**



##### 1.3.4 Other training configs

In addition to the experiments above, I also tried different training configs, like using different optimizer, scheduler, batch size and training epochs.

**(1) optimizer**

I tried to use SGD with different learning rate, momentum and weight decay. In addition, I also attempted to leverage Adam optimizer. Results are shown below.

| optimizer | learning rate | momentum | weight decay | Accuaracy   |
| --------- | ------------- | -------- | ------------ | ----------- |
| **SGD**   | 0.1           | 0.9      | 1e-4         | 88.2700     |
| **SGD**   | 0.1           | 0.9      | 1e-3         | **91.0200** |
| **SGD**   | 0.1           | 0.8      | 1e-3         | 90.7600     |
| **SGD**   | 0.01          | 0.9      | 1e-3         | 89.9500     |
| **Adam**  | /             | /        | /            | 90.8300     |

**(2) loss function**

Label smoothing is to transform one-hot labels into soft labels, which aims at improving the generalization ability of networks. In this project, I tried cross entropy with and without label smoothing. Results are shown below:

| Loss function               | Accuracy    |
| --------------------------- | ----------- |
| **without label_smoothing** | 91.0200     |
| **label_smoothing=0.1**     | 91.3000     |
| **label_smoothing=0.2**     | **91.6400** |
| **label_smoothing=0.3**     | 91.3800     |

**(3) activation function**

In addition to ReLU, I also tried Leaky ReLU and Swish as activation function. Results are shown below:

| Activations          | Accuracy    |
| -------------------- | ----------- |
| **ReLU**             | **91.6400** |
| **Leaky ReLU(0.01)** | 91.3700     |
| **Swish**            | 89.5700     |

**(4) batch size**

To figure out whether batch size will influence the accuracy, I tried different batch size ranging from 16 to 256. Results are shown below:

| batch size | Accuracy    | Training time(s) |
| ---------- | ----------- | ---------------- |
| **16**     | 86.5200     | 596.97           |
| **32**     | 88.9000     | 329.27           |
| **64**     | 90.6900     | 231.96           |
| **128**    | **91.6400** | 236.45           |
| **256**    | 88.9000     | 233.83           |
| **512**    | 86.1600     | 238.53           |

Theoretically, smaller batch size will introduce more noise, thus influences the convergence. However, it's easier for smaller batch size to jump out of local minimum, leading to better generalization ability. Besides, larger batch size requires larger memory, but may cost less time due to parallelism. According to experiments, the best batch size for CIFAR-10 is 128.

**(5) training epochs**

At last, I expanded the training time to see whether longer training time definitely brings better results. I trained the best model for 30, 50 and 100 epochs without early stopping. Results are shown below:

| Epochs | Accuracy | Training time(s) |
| ------ | -------- | ---------------- |
| 30     | 91.6400  | 236.45           |
| 50     | 93.1000  | 387.40           |
| 100    | 93.2400  | 793.62           |

Training curve of 100 epochs:

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\epoch_100.png" alt="epoch_100" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 7: Training curves of 100 epochs </b></center>

According to the experiment, training 100 epochs only improved the accuracy for about $0.14\%$ , while costs a doubled training time compared to training 50 epochs, also causing the risk of overfitting. For this model, training for 50 epochs is good enough.



 #### 1.4 Results

In summary, I adjusted the original ResNet-18 network and tried different regularization methods and optimization strategies. Finally, my model achieved $91.64\%$ accuracy after training 30 epochs and $93.10\%$ after 50 epochs. Longer training time has been proved to gain little improvement. The model uploaded to Google drive is the one after training 50 epochs.



#### 1.5 Reproduction of "Deep Residual Learning for Image Recognition"(ResNet-20)

Occasionally, I found that the researchers of ResNet has already adapted their networks for CIFAR-10 and conducted some experiments. They trained a series of networks with various depth and achieved $90\%$ to $95\%$ accuracy on CIFAR-10. Among the networks they used, I tried to construct the 20-layers ResNet (so-called ResNet-20) and trained it on CIFAR-10 to see whether their adjustment take effect. Compared to ResNet-18, ResNet-20 cuts down the number of channels and deepens the net by expanding each composed layer to have 3 pairs of convolution layers. The architecture of the network is shown in **Figure 8**:

<img src="D:\学习\大三\下学期\神经网络\PJ2\resnet20.drawio (1).png" alt="resnet20.drawio (1)" style="zoom:67%;" />

<center style="font-size:16px"><b>Figure 8: The architecture of ResNet-20 by ResNet team</b></center>

Their training configs are shown below:

``` python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduluer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
epochs=180
```

In fact, the configs shown here are not exactly the same as theirs due to some difficulty of practice, but very close.

Training results:

```
[Train]Total training time:705.2488746643066
[Train]Training epochs:180
[Test]Accuracy:91.3000
```

The results of this experiment（$91.30\%$） are very close to theirs（$91.25\%$）, so it is a successful reproduction.

Training curve:

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task1\imgs\res20.png" alt="res20" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 9: Training curves of ResNet-20</b></center>



### Task2: BatchNorm

#### 2.1 Introduction

Batch Normalization (BatchNorm) is a widely adopted technique that enables faster and more stable training of deep neural networks. It is a popular belief that BatchNorm optimizes the training process by controlling the distribution of the input data during the forward process. This task aims at gaining a deeper insight into the mechanism behind BatchNorm by comparing the loss landscape of VGG network with and without BatchNorm layer.

#### 2.2 Experiments

The first problem is, how to add BatchNorm Layer into VGG? It's popular to add BatchNorm layer after convolution layers, but how about full connected layer? To figure out that, I tried two different strategies:

**Strategy 1:** Only add BatchNorm2d layer after convolution layers.

**Strategy 2:** On the basis of strategy 1, add BatchNorm1d layer after full connected layers in the classifier (except for the last one).

For each strategy, I trained the original VGG and VGG with BatchNorm for 20 epochs using various learning rates among `[1e-3, 2e-3, 5e-4, 1e-4]`, logged the min and max losses of each step and plot them with *matplotlib.pyplot*. Additionally, I found that setting batch size as 128 will lead to great oscillation of loss, so I also tried batch size as 256 to eliminate noise. 

Since the purpose of this task is to figure out the effect of BatchNorm, I didn't save any weights of models during the experiment.

#### 2.3 Results

##### 2.3.1 Training curves

**Figure 10** shows the training curves of original VGG net, VGG_BatchNorm with strategy 1, and VGG_BatchNorm with strategy 2. (batch size=128, learning rate=1e-3)

---

![curves](D:\学习\大三\下学期\神经网络\PJ2\codes\Task2\VGG_Batchnorm\vis\curves.png)

<center style="font-size:16px"><b>Figure 10: Training curves of VGG with and without BatchNorm</b></center>

---

It's hard to proof that BatchNorm has an significant influence on loss landscape despite the training curves plotted, so I follow the method to visualize the loss landscape by plotting the minimum and maximum of losses for each step in 2.3.2. 

##### 2.3.2 Loss Landscape

**Figure 11, 12** demonstrates the loss landscape of VGG with and without BatchNorm. **Figure 11** shows the case when batch size is 128, whie **Figure 12** shows the case when batch size is 256.

---

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task2\VGG_Batchnorm\vis\bs_128.png" alt="bs_128" style="zoom: 40%;" />

<center style="font-size:16px"><b>Figure 11: Loss landscape of VGG with with different BatchNorm(batch size=128)</b></center>

---

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task2\VGG_Batchnorm\vis\bs_256.png" alt="bs_256" style="zoom: 40%;" />

<center style="font-size:16px"><b>Figure 12: Loss landscape of VGG with with different BatchNorm(batch size=256)</b></center>

---

According to the experiments, BatcnNorm shows its capability in smoothing the loss landscape and making training process more stable, especially with **strategy 2** (i.e. add BatchNorm not only after convolution layers, but also after full connected layer). To be more intuitively, I also plot the difference between the max loss and the min loss of each step for the original VGG and VGG_BatchNorm with strategy 2 in **Figure 13**.

<img src="D:\学习\大三\下学期\神经网络\PJ2\codes\Task2\VGG_Batchnorm\vis\v3_compare.png" alt="v3_compare" style="zoom:72%;" />

<center style="font-size:16px"><b>Figure 13:Range of loss for VGG and VGG_BN</b></center>
