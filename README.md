# 1.项目背景介绍
新冠肺炎目前的死亡率已降至2%左右，随着各国新冠疫苗接种率不断提高，死亡率还会进一步下降，所以，站在全世界层面分析，新冠肺炎病毒与人类长期共存的可能性很大。搜索资料后发现，新冠病毒感染者的CT影像学特征表现为：会出现一些磨玻璃影或者是浸润影。国家卫健委出台的几版的诊疗方案中都把CT影像学检查作为诊断标准之一，通过计算机视觉的方法可以对患者的CT影像进行判断，可以提高医生对患者病情判断的正确率，加快CT结果鉴别速度，助力新冠疫情抗疫工作。

# 2.数据介绍
* 数据描述：来自卡塔尔多哈卡塔尔大学和孟加拉国达卡大学的一组研究人员，以及来自巴基斯坦和马来西亚的合作者与医生合作，建立了一个针对COVID-19阳性病例的胸部X射线图像数据库，以及正常和病毒性肺炎图像。
* 数据说明：本数据是一个COVID-19阳性病例的胸部X射线图像以及正常和病毒性肺炎图像的数据库。 数据包含有1200个COVID-19阳性图像，1341正常图像和1345病毒性肺炎图像。
* 源地址：https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

## 解压数据


```python
!mkdir /home/aistudio/data/images
!unzip -q /home/aistudio/data/data34241/covid19-combo.zip -d /home/aistudio/data/images 
!mv /home/aistudio/data/images/'COVID-19 Radiography Database'/* /home/aistudio/data/images
!rm -rf /home/aistudio/data/images/'COVID-19 Radiography Database'
!ls ~/data/images
```

    COVID-19		NORMAL.metadata.xlsx  Viral Pneumonia.matadata.xlsx
    COVID-19.metadata.xlsx	README.md.txt
    NORMAL			Viral Pneumonia


此时可以看到在 ~/data/images 目录下有三个文件夹，COVID-19,Viral Pneumonia和NORMAL，分别存放着三个类别的图像。除了训练图片，PaddleClas还需要我们提供一个数据列表文件，里面每条数据按照 “文件路径  类别” 的格式标记，以供后续训练。


```python
import os
base_dir = "/home/aistudio/data/images/" # CT图片所在路径
img_dirs = ["COVID-19", "NORMAL", "Viral Pneumonia"] # 三类CT图片文件夹名

file_names = ["train_list.txt", "val_list.txt", "test_list.txt"]
splits = [0, 0.6, 0.8, 1] # 按照 6 2 2 的比例对数据进行分组

for split_ind, file_name in enumerate(file_names):
    with open(os.path.join("./data", file_name), "w") as f:
        for type_ind, img_dir in enumerate(img_dirs):
            imgs = os.listdir(os.path.join(base_dir, img_dir) )
            for ind in range( int(splits[split_ind]* len(imgs)), int(splits[split_ind + 1] * len(imgs)) ):
                print("{}|{}".format(img_dir + "/" + imgs[ind], type_ind), file = f)
```

文件列表制作完成，我们可以用head查看一下前10行。


```python
! head /home/aistudio/data/train_list.txt
```

    COVID-19/COVID-19(197).png|0
    COVID-19/COVID-19(191).png|0
    COVID-19/COVID-19 (72).png|0
    COVID-19/COVID-19 (52).png|0
    COVID-19/COVID-19 (48).png|0
    COVID-19/COVID-19 (104).png|0
    COVID-19/COVID-19(151).png|0
    COVID-19/COVID-19 (121).png|0
    COVID-19/COVID-19 (122).png|0
    COVID-19/COVID-19 (113).png|0


# 3.PaddleClas介绍
飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

模型主要分为两类：服务器端模型和移动端模型。移动端模型以轻量化为主要设计目标，通常速度快体积小，但是会牺牲一定的精度。我们在这个项目中选择服务器端模型，并最终选择了ResNet_Vd。这个选择主要是考虑到项目的数据量不是很大，其他基于ResNet的模型ResNeXt，SENet和Res2Net都一定程度上增加了模型的参数量，这个数据量可能不足以支撑训练。HRnet主要是针对细节特征有优势，不是很符合我们的场景而且参数量也不小。ResNet_Vd是ppcls框架主推的模型，经过了大量精度和速度上的优化。

图像分类模型有大有小，其应用场景各不相同，在云端或者服务器端应用时，一般情况下算力是足够的，更倾向于应用高精度的模型；在手机、嵌入式等端侧设备中应用时，受限于设备的算力和内存，则对模型的速度和大小有较高的要求。PaddleClas同时提供了服务器端模型与端侧轻量化模型来支撑不同的应用场景。
![](https://ai-studio-static-online.cdn.bcebos.com/c2f8c56b674f48448764775681a4c91a721354156a4c497caa1365d75cec72e9)


下面我们获取开发套件代码。为了加快速度，使用gitee的克隆，在终端使用命令

git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.1


# 4.模型训练

首先查看配置文件


```python
!cat ~/covid.yaml
```

    mode: 'train'
    ARCHITECTURE:
        # 使用的模型结构，可以参照 pdclas/config 下其他模型结构的cofig文件修改模型名称
        # 比如 ResNet101
        name: 'ResNet50_vd'
    pretrained_model: "" # 通常使用预训练模型迁移能在小数据集上取得好的效果，但是预训练模型都是针对自然图像，因此没有使用
    model_save_dir: "./output/"
    classes_num: 3
    total_images: 2905
    save_interval: 1
    validate: True
    valid_interval: 1
    epochs: 20 
    topk: 2
    image_shape: [3, 1024, 1024]


​    
    LEARNING_RATE:
        function: 'Cosine'    
        params:                   
            lr: 0.00375
    
    OPTIMIZER:
        function: 'Momentum'
        params:
            momentum: 0.9
        regularizer:
            function: 'L2'
            factor: 0.000001
    
    TRAIN:
        batch_size: 4 # 训练过程中一个batch的大小，如果你有幸分到32g显卡这个参数最高开到16
        num_workers: 4
        file_list: "/home/aistudio/data/train_list.txt"
        data_dir: "/home/aistudio/data/images/"
        delimiter: "|"
        shuffle_seed: 0
        transforms:
            - DecodeImage:
                to_rgb: True
                to_np: False
                channel_first: False
            - RandFlipImage:
                flip_code: 1
            - ToCHWImage:
    
    VALID:
        batch_size: 20
        num_workers: 4
        file_list: "/home/aistudio/data/val_list.txt"
        data_dir: "/home/aistudio/data/images/"
        delimiter: "|"
        shuffle_seed: 0
        transforms:
            - DecodeImage:
                to_rgb: True
                to_np: False
                channel_first: False
            - ResizeImage:
                resize_short: 1024
            - ToCHWImage:


ResNet50有两个基本的块，分别名为Conv Block和Identity Block，其中Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度；Identity Block输入维度和输出维度相同，可以串联，用于加深网络的。PaddleClas已经内置写好的ResNet50,无需我们手动写。

接下来就可以开始训练了，一般pdclas是在命令行环境下使用的，这里需要注意的一点是启动训练之前需要设置一个环境变量。命令行启动训练代码如下。

```shell
cd ~/pdclas
export PYTHONPATH=./:$PYTHONPATH
python -m paddle.distributed.launch --selected_gpus="0" tools/train.py -c ../covid.yaml 
```
在notebook中可以用os.environ设置环境变量，下面的block会运行训练代码，和上面的三行等效。训练需要的时间比较长，为了方便大家浏览后面的步骤，提供的config中只训练了一个epoch，大概3分钟。想要自己训练一个效果更好的网络可以修改 ~/covid.yaml 文件，大概训练15个epoch能得到97%精度的模型。


```python
%cd ~/PaddleClas/
import os 
os.environ['PYTHONPATH']="/home/aistudio/PaddleClas"
!python -m paddle.distributed.launch --selected_gpus="0" tools/train.py -c ../covid.yaml 
```

    2022-02-27 07:36:51 INFO: epoch:8   train step:413  loss:  0.2000 top1: 1.0000 top2: 1.0000 lr: 0.003201 elapse: 0.267s

在笔者训练的过程中测试集 Top 1 准确率最高达到了 **97%**，在这个数据集上最高报告的 Top 1 准确率是 98%，**达到了SOTA效果**，这也证明了ppcls开发套件强大的实力。

训练出满意的模型后我们会希望用训练得到的模型进行前向推理，这需要用到训练过程中保存的权重文件。Paddle框架保存的权重文件分为两种：支持前向推理和反向梯度的训练模型 和 只支持前向推理的推理模型。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。ppcls在训练过程中保存的模型属于训练模型，默认保存在 **~/pdclas/output/模型结构** 路径下。


```python
!ls ~/PaddleClas/output/ResNet50_vd
```

我们通过pdclas中提供的模型转换脚本将训练模型转换为推理模型


```python
!python tools/export_model.py --m=ResNet50_vd --p=output/ResNet50_vd/best_model_in_epoch_0/ppcls --o=../inference
!ls -lh /home/aistudio/inference/
```

可以看到转换之后生成了两个文件，model是模型结构，params是模型权重。转换完毕，最后一步是进行推理，笔者已经保存了一个训练了15个epoch的模型，可以直接使用。如果你希望使用自己的模型，按照上面的步骤对自己训练的模型进行转换后修改下方模型和权重的路径就可以。推理的输入图片可以从 **/home/aistudio/data/images** 路径下的三个图片文件夹中任意选择，填入对应的路径即可。这里我们demo使用的是项目开始时展示的新冠CT图片。

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/00faf7de665d4eb1b73b4adcaba2c7e8bd800f5cf9284ddbba627c5a6b087b99" style="zoom:40%;" >
</div>


```python
!python tools/infer/predict.py --use_gpu=0 -i="/home/aistudio/COVID-19 (2).png"     -m=/home/aistudio/pretrain/model     -p=/home/aistudio/pretrain/params 
```

分类结果：新冠类别为0，正常类别为1，其他肺炎类别为2

以上代码参考自：https://aistudio.baidu.com/aistudio/projectdetail/463184?channelType=0&channel=0

# 5.本文总结
这是我的第一次使用PaddleClas组件做项目，虽然借鉴了别人的经验，但还是收获满满。飞浆对这些模型套件的开发大大的降低了使用者的使用难度，并且也保证了非常高的准确度，我认为开发各种套件这样的工作是非常有意义的，剩下的就是使用者使用套件进行创新。

# 6.个人总结
本人目前就读于河南理工大学计算机科学与技术学院，正值大三考研时刻（2023），并且还进行着人工智能-推荐系统方向的研究。

AI Studio个人主页：https://aistudio.baidu.com/aistudio/usercenter
