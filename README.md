项目的AiStudio地址：https://aistudio.baidu.com/aistudio/projectdetail/3532243

# 1.项目背景介绍

新冠肺炎目前的死亡率已降至2%左右，随着各国新冠疫苗接种率不断提高，死亡率还会进一步下降，所以，站在全世界层面分析，新冠肺炎病毒与人类长期共存的可能性很大。搜索资料后发现，新冠病毒感染者的CT影像学特征表现为：会出现一些磨玻璃影或者是浸润影。国家卫健委出台的几版的诊疗方案中都把CT影像学检查作为诊断标准之一，通过计算机视觉的方法可以对患者的CT影像进行判断，可以提高医生对患者病情判断的正确率，加快CT结果鉴别速度，助力新冠疫情抗疫工作。

# 2.数据介绍
* 数据描述：来自卡塔尔多哈卡塔尔大学和孟加拉国达卡大学的一组研究人员，以及来自巴基斯坦和马来西亚的合作者与医生合作，建立了一个针对COVID-19阳性病例的胸部X射线图像数据库，以及正常和病毒性肺炎图像。不同的是，该数据集目录名和文件名进行了去空格重新编号处理。
* 数据说明：本数据是一个COVID-19阳性病例的胸部X射线图像以及正常和病毒性肺炎图像的数据库。 数据包含有1200个COVID-19阳性图像，1341正常图像和1345病毒性肺炎图像。

AiStudio地址：https://aistudio.baidu.com/aistudio/datasetdetail/105737      
源数据地址：https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

## 解压数据


```python
# 解压所挂载的数据集到data下
!unzip -oq data/data105737/Database.zip -d data
```


```python
# 查看数据集的目录结构
! tree /home/aistudio/data/Database -d
```

    /home/aistudio/data/Database
    ├── COVID19
    ├── NORMAL
    └── ViralPneumonia
    
    3 directories


此时可以看到在 ~/data/Database 目录下有三个文件夹，COVID19,ViralPneumonia和NORMAL，分别存放着三个类别的图像。除了训练图片，PaddleClas还需要我们提供一个数据列表文件，里面每条数据按照 “文件路径  类别” 的格式标记，以供后续训练。


```python
import os
base_dir = "/home/aistudio/data/Database/" # CT图片所在路径
img_dirs = ["COVID19", "NORMAL", "ViralPneumonia"] # 三类CT图片文件夹名

file_names = ["train_list.txt", "val_list.txt", "test_list.txt"]
splits = [0, 0.6, 0.8, 1] # 按照 6 2 2 的比例对数据进行分组

for split_ind, file_name in enumerate(file_names):
    with open(os.path.join("/home/aistudio/data", file_name), "w") as f:
        for type_ind, img_dir in enumerate(img_dirs):
            imgs = os.listdir(os.path.join(base_dir, img_dir) )
            for ind in range( int(splits[split_ind]* len(imgs)), int(splits[split_ind + 1] * len(imgs)) ):
                print("{} {}".format(img_dir + "/" + imgs[ind], type_ind), file = f)
```

文件列表制作完成，我们可以用head查看一下前10行。


```python
! head /home/aistudio/data/train_list.txt
```

    COVID19/COVID147.png 0
    COVID19/COVID109.png 0
    COVID19/COVID120.png 0
    COVID19/COVID003.png 0
    COVID19/COVID035.png 0
    COVID19/COVID114.png 0
    COVID19/COVID005.png 0
    COVID19/COVID156.png 0
    COVID19/COVID174.png 0
    COVID19/COVID081.png 0



```python
# 显示三种CT图像
%matplotlib inline
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
covid="/home/aistudio/data/Database/COVID19/COVID002.png"
normal="/home/aistudio/data/Database/NORMAL/Normal002.png"
viral="/home/aistudio/data/Database/ViralPneumonia/Viral002.png"
img1=mpimg.imread(covid)
img2=mpimg.imread(normal)
img3=mpimg.imread(viral)
plt.subplot(131)
plt.imshow(img1)
plt.subplot(132)
plt.imshow(img2)
plt.subplot(133)
plt.imshow(img3)
plt.show()
```


​    ![](https://gitee.com/JIANG-HS/myphotos/raw/master/photos/output_8_0.png)
​    


# 3.PaddleClas介绍
飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

在 PaddleClas 中，图像识别，是指给定一张查询图像，系统能够识别该查询图像类别。广义上，图像分类也是图像识别的一种。但是与普通图像识别不同的是，图像分类只能判别出模型已经学习的类别，如果需要添加新的类别，分类模型只能重新训练。PaddleClas 中的图像识别，对于陌生类别，只需要更新相应的检索库，就能够正确的识别出查询图像的类别，而无需重新训练模型，这大大增加了识别系统的可用性，同时降低了更新模型的需求，方便用户部署应用。

对于一张待查询图片，PaddleClas 中的图像识别流程主要分为三部分：

1. 主体检测：对于给定一个查询图像，主体检测器首先检测出图像的物体，从而去掉无用背景信息，提高识别精度。
1. 特征提取：对主体检测的各个候选区域，通过特征模型，进行特征提取
1. 特征检索：将提取的特征与特征库中的向量进行相似度比对，得到其标签信息

图像分类模型有大有小，其应用场景各不相同，在云端或者服务器端应用时，一般情况下算力是足够的，更倾向于应用高精度的模型；在手机、嵌入式等端侧设备中应用时，受限于设备的算力和内存，则对模型的速度和大小有较高的要求。PaddleClas同时提供了服务器端模型与端侧轻量化模型来支撑不同的应用场景。

![](https://ai-studio-static-online.cdn.bcebos.com/5d4a2a1a2cf548e29bd8d47eb2588ed442b14928fc734ae4b96612ae20e6ba08)


下面我们**获取开发套件代码**。为了加快速度，使用gitee的克隆，在终端使用命令

```
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.3
```




# 4.模型训练

首先查看使用ResNeXt101_64x4d的配置文件


```python
!cat ~/covid.yaml
```

    # global configs
    Global:
      checkpoints: null
      pretrained_model: null
      output_dir: ./output/
      device: gpu
      save_interval: 1
      eval_during_train: True
      eval_interval: 1
      epochs: 20
      print_batch_step: 10
      use_visualdl: False
      # used for static mode and model export
      image_shape: [3, 224, 224]
      save_inference_dir: ./inference
    
    # model architecture
    Arch:
      name: ResNeXt101_64x4d
      class_num: 2905
     
    # loss function config for traing/eval process
    Loss:
      Train:
        - CELoss:
            weight: 1.0
      Eval:
        - CELoss:
            weight: 1.0
    
    Optimizer:
      name: Momentum
      momentum: 0.9
      lr:
        name: Piecewise
        learning_rate: 0.1
        decay_epochs: [30, 60, 90]
        values: [0.1, 0.01, 0.001, 0.0001]
      regularizer:
        name: 'L2'
        coeff: 0.00015
    
    # data loader for train and eval
    DataLoader:
      Train:
        dataset:
          name: ImageNetDataset
          image_root: /home/aistudio/data/Database
          cls_label_path: /home/aistudio/data/train_list.txt
          transform_ops:
            - DecodeImage:
                to_rgb: True
                channel_first: False
            - RandCropImage:
                size: 224
            - RandFlipImage:
                flip_code: 1
            - NormalizeImage:
                scale: 1.0/255.0
                mean: [0.485, 0.456, 0.406]
                std: [0.229, 0.224, 0.225]
                order: ''
    
        sampler:
          name: DistributedBatchSampler
          batch_size: 64
          drop_last: False
          shuffle: True
        loader:
          num_workers: 4
          use_shared_memory: True
    
      Eval:
        dataset: 
          name: ImageNetDataset
          image_root: /home/aistudio/data/Database
          cls_label_path: /home/aistudio/data/val_list.txt
          transform_ops:
            - DecodeImage:
                to_rgb: True
                channel_first: False
            - ResizeImage:
                resize_short: 256
            - CropImage:
                size: 224
            - NormalizeImage:
                scale: 1.0/255.0
                mean: [0.485, 0.456, 0.406]
                std: [0.229, 0.224, 0.225]
                order: ''
        sampler:
          name: DistributedBatchSampler
          batch_size: 64
          drop_last: False
          shuffle: False
        loader:
          num_workers: 4
          use_shared_memory: True
    
    Infer:
      infer_imgs: /home/aistudio/COVID002.png
      batch_size: 10
      transforms:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
      PostProcess:
        name: Topk
        topk: 3
        class_id_map_file: ppcls/utils/imagenet1k_label_list.txt
    
    Metric:
      Train:
        - TopkAcc:
            topk: [1, 5]
      Eval:
        - TopkAcc:
            topk: [1, 5]


ResNet50有两个基本的块，分别名为Conv Block和Identity Block，其中Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度；Identity Block输入维度和输出维度相同，可以串联，用于加深网络的。PaddleClas已经内置写好的ResNet50,无需我们手动写。

接下来就可以开始训练了，首先必须进入PaddleClas根目录；运行命令中，-c 用于指定配置文件的路径。

需要注意的是建议**使用至尊版GPU运行**，不然可能会有错误


```python
%cd ~/PaddleClas/
# 进入PaddleClas根目录，执行此命令
!python tools/train.py -c /home/aistudio/covid.yaml
```


这里只训练了20个epoch，Top 1 准确率最最好结果达到了 **86%**
，读者也可以自行增加训练次数以得到更高的准确率。


# 5.模型评估



```python
%cd ~/PaddleClas/
!python3 tools/eval.py \
    -c /home/aistudio/covid.yaml    \
    -o Global.pretrained_model=./output/ResNeXt101_64x4d/best_model
```

    W0227 17:48:37.934310  2161 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [2022/02/27 17:48:45] root INFO: [Eval][Epoch 0][Iter: 0/10]CELoss: 1.38086, loss: 1.38086, top1: 0.53125, top5: 1.00000, batch_cost: 2.59758s, reader_cost: 2.35900, ips: 24.63829 images/sec
    [2022/02/27 17:48:48] root INFO: [Eval][Epoch 0][Avg]CELoss: 0.48312, loss: 0.48312, top1: 0.83133, top5: 1.00000


# 6.模型推理
通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理： 首先，对训练好的模型进行转换：


```python
%cd ~/PaddleClas/
!python3 tools/export_model.py \
    -c /home/aistudio/covid.yaml \
    -o Global.pretrained_model=output/ResNeXt101_64x4d/best_model
```

```python
inf="/home/aistudio/data/Database/COVID19/COVID111.png"
img=mpimg.imread(inf)
plt.imshow(img1)
plt.show()
```


![](https://gitee.com/JIANG-HS/myphotos/raw/master/photos/output_22_0.png)

    

```python
%cd ~/PaddleClas/deploy/

!python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=/home/aistudio/data/Database/COVID19/COVID111.png \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None
```

分类结果：
```
COVID111.png:	class id(s): [0, 2, 1, 2659, 704], score(s): [0.63, 0.30, 0.07, 0.00, 0.00]
```
新冠类别为0，正常类别为1，其他肺炎类别为2

排在首位的是类别0，推测结果准确！


# 7.本文总结
这是我的第一次使用PaddleClas组件做项目，虽然借鉴了别人的经验，但还是收获满满。飞浆对这些模型套件的开发大大的降低了使用者的使用难度，并且也保证了非常高的准确度，我认为开发各种套件这样的工作是非常有意义的，剩下的就是使用者使用套件进行创新。

# 8.个人总结
本人目前就读于河南理工大学计算机科学与技术学院，正值大三考研时刻（2023），并且还进行着人工智能-推荐系统方向的研究。

AI Studio个人主页：https://aistudio.baidu.com/aistudio/usercenter
