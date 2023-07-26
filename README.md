# LAMP
This repo is the official implementation for: ["LAMP: Leveraging Language Prompts for Multi-person Pose Estimation"] (https://arxiv.org/abs/2307.11934) @IROS2023

## Method
![image](https://github.com/shengnanh20/LAMP/blob/main/lamp.png)


## Requirements

* python 3.9
* pytorch 1.12.1
* torchvision 0.13.1
* clip 1.0

## Datasets

* [OCHuman](https://github.com/liruilong940607/OCHumanApi)
* [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)

## Training

* To train on OCHuman from ImageNet pretrained models with multiple gpus, you can run: 
```
python3 tools/train.py --cfg experiments/ochuman.yaml --gpus 0,1,2,3
```
where --cfg indicates the configure file and --gpus implys the numers of gpus.
You can replace the configure file for other datasets.

## Testing

* To test the model which has been trained on the OCHuman dataset, you can run the testing script as following:
```
python tools/valid.py --cfg experiments/ochuman.yaml --gpus 0,1,2,3 TEST.MODEL_FILE MODEL_PATH/model_best.pth.tar
```
Replace MODEL_PATH with your local path of the trained model.

## Citation

Please cite the following paper if you find this repository useful in your research.
```
@misc{hu2023lamp,
      title={LAMP: Leveraging Language Prompts for Multi-person Pose Estimation}, 
      author={Shengnan Hu and Ce Zheng and Zixiang Zhou and Chen Chen and Gita Sukthankar},
      year={2023},
      eprint={2307.11934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Reference
The code is mainly encouraged by
* [CID](https://github.com/kennethwdk/CID)
* [MMPose](https://github.com/open-mmlab/mmpose)
* [CLIP](https://github.com/openai/CLIP)
