# model
- vggnet version1
- alexnet

## reuqirements
- keras 2.1
- ubuntu 16.04 
- tensorflow-gpu

## train

- NI alexnet
```
python train_alexnet_NI.py -dp your_data_path
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