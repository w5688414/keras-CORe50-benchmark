# model
- vgg16
- alexnet

## reuqirements
- keras 2.1.4
- ubuntu 16.04 
- tensorflow-gpu 1.4
- python3

## train

1. download the dataset

2. download caffe source code
```
git clone https://github.com/vlomonaco/core50
```
3. train

- NI alexnet
```
python train_alexnet_NI.py -dp your_data_path
```

- NI vgg16
```
python train_vgg16_NI.py -dp your_data_path
```

- NC vgg16
```
python train_vgg16_NC.py -dp your_data_path
```

- NC alexnet
```
python train_alexnet_NC.py -dp your_data_path
```

- NIC alexnet
```
python train_alexnet_NIC.py -dp your_data_path
```


- NIC vgg16
```
python train_vgg16_NIC.py -dp your_data_path
```

## reference
[https://github.com/chenlongzhen/vgg_finetune](https://github.com/chenlongzhen/vgg_finetune)