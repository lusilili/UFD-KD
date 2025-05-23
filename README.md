# Unified Frequency Decoupling Knowledge Distillation for Computer Vision

![image](https://github.com/user-attachments/assets/cd473ee2-8356-4925-abb9-064c4dbc0eac)

Here are some visulizations for the feature distribution in frequency domain (ViT-large and ResNet18):

<img src="https://github.com/user-attachments/assets/5d2dba44-f605-4163-8d33-46cbd5509aba" width="500px">
<img src="https://github.com/user-attachments/assets/eb753fed-6b29-4a92-a31f-68ae5813add2" width="600px">

Showing the mismatch of feature distribution across models, which is also the main problem our method tries to resolve, we proposed the UFD framework that decouples the feature in both spatial and channel dimensions to achieve better feature alignment in knowledge distillation.

The strategy we apply for decoupling is quite simple which is based on the basic properties of Frequency domain(DC Components and AC Components). The process is shown as follows:
<img src="https://github.com/user-attachments/assets/3f4489bd-aab2-4080-977c-fb8cf9905823" width="700px">

This repository includes official implementation for the following papers:



* ICCV 2023: [NKD and USKD](https://github.com/yzd-v/cls_KD/blob/1.0/nkd.md): From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels

* NeurIPS 2024: [ScaleKD](https://github.com/deep-optimization/ScaleKD): Strong Vision Transformers Could Be Excellent Teachers

* [ViTKD](https://github.com/yzd-v/cls_KD/blob/1.0/vitkd.md): Practical Guidelines for ViT feature knowledge distillation

It also provides unofficial implementation for the following papers:
* [KD](https://arxiv.org/abs/1503.02531), [DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), [WSLD](https://arxiv.org/abs/2102.00650)
* [MGD](https://arxiv.org/abs/2205.01529), [SRRL](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/70425/Tzimiropoulos%20Knowledge%20distillation%20via%202021%20Accepted.pdf?sequence=2)

If this repository is helpful, please give us a star ⭐ and cite relevant papers.

## Install
  - Prepare the dataset in data/imagenet AND data/cifar, also the pretrained teacher checked point will be released soon on google drive.
    · Following this repository,
    · Download the ImageNet dataset from http://www.image-net.org/.
    · Then, move and extract the training and validation images to labeled subfolders, using the following script.
    · Move the data into folder data/imagenet
  - ```
    # Set environment
    · Python 3.8 (Anaconda is recommended)
    · CUDA 11.1
    · PyTorch 1.10.1
    · Torchvision 0.11.2

    # create conda environment
    conda create -n openmmlab python=3.8
    # enter the environment
    conda activate openmmlab
    # install packages
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
    pip install -r requirements.txt
    ```
  - This repo uses mmcls = 1.0.0rc6. If you want to use lower mmcls version for distillation, you can refer branch [master](https://github.com/yzd-v/cls_KD/tree/master) to change the codes.

  - To validate the UFD-KD's capability of different vision task, we transfer this method from image classification to semantic segmentation (using the basic training framework from Af-DCD and its results as baseline), this will be included in the code base soon.
  - 

## Run
  - The experiments based on the traditional training strategy are performed on 8 GPUs from a single node.

    Training configurations for various teacher-student network pairs are in folder <configs/distillers/>

  - Run distillation by following command:

    bash tools/dist_train.sh $CONFIG_PATH $NUM_GPU
    for example: bash tools/dist_train.sh configs/distillers/traditional_traning_strategy/swin-s_distill_res50_img_s3_s4.py 8



## Citing NKD and USKD
```
@article{yang2023knowledge,
  title={From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels},
  author={Yang, Zhendong and Zeng, Ailing and Li, Zhe and Zhang, Tianke and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2303.13005},
  year={2023}
}
```

## Citing ViTKD
```
@article{yang2022vitkd,
  title={ViTKD: Practical Guidelines for ViT feature knowledge distillation},
  author={Yang, Zhendong and Li, Zhe and Zeng, Ailing and Li, Zexian and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2209.02432},
  year={2022}
}
```

## Citing OFAKD
```
@inproceedings{hao2023ofa,
  author    = {Zhiwei Hao and Jianyuan Guo and Kai Han and Yehui Tang and Han Hu and Yunhe Wang and Chang Xu},
  title     = {One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

## Citing ScaleKD
```
@article{fan2024scalekd,
  title={ScaleKD: Strong Vision Transformers Could Be Excellent Teachers},
  author={Fan, Jiawei and Li, Chao and Liu, Xiaolong and Yao, Anabang},
  journal={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Citing Af-DCD
```
@inproceedings{fan2023augmentation,
  title={Augmentation-free Dense Contrastive Knowledge Distillation for Efficient Semantic Segmentation},
  author={Fan, Jiawei and Li, Chao and Liu, Xiaolong and Song, Meina and Yao, Anbang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgement

Our code is based on the project [MMPretrain](https://github.com/open-mmlab/mmpretrain/tree/main) and [cls_KD repository](https://github.com/yzd-v/cls_KD).
We thank the authors of the two repositories for releasing their amazing codes.
