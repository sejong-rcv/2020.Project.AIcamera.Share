# Pseudo-Lidar

## Dataset

### KAIST AAAI 2017

실험한 알고리즘은 학습 시에는 Stereo-pair images와 테스트 시에는 single image가 필요하다.
학습과 테스트는  KAIST AAAI2017 데이터 셋을 사용했다.
학습에는 3037 장,테스트에는 1784 장이 포함되어있다. 

## Dataloader
Dataloader는 폴더의 다음 구조를 가정합니다 ("data_dir" 에는 Kaist_data이 들어가야한다.)
왼쪽 영상의 경우 LEFT 폴더 속에있고 , 오른쪽 영상의 경우 RIGTH에 있으니 찾기 쉬울 것이다.
***그리고 데이터를 불러오기 위한 txt 파일은 Kaist에서 제공되는 것이 아니니 ```cp``` 나 ```mv```를 이용해 txt를 옳겨줘야한다. ***

예 ) 데이터 폴더 구조 (이 예에서는 "Kaist_data" 디렉토리 경로를 'data_dir' 로 전달해야 한다 ) :
```
data
├── Kaist_data
│   ├── training
│   │   ├── Campus
│   │   │   ├─ DEPTH
│   │   │   │   ├── DEPTH_000000000.mat
│   │   │   │   └── ...
│   │   │   ├─ LEFT
│   │   │   │   ├── LEFT_000000000.jpg
│   │   │   │   └── ...
│   │   │   ├─ RIGTH
│   │   │   │   ├── RIGHT_000000000.jpg
│   │   │   │   └── ...
│   │   │   ├─ THERMAL
│   │   │   │   ├── THERMAL_000000000.jpg
│   │   │   │   └── ...
│   │   ├── Urban
│   │   │   ├── DEPTH
│   │   │   │   ├── DEPTH_000000000.mat
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── testing
│   │   ├── Campus
│   │   │   ├─ DEPTH
│   │   │   └── ...
│   │   └── ...
│   ├── txt
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── test_depth.txt
│   └── ...
├── models
├── output

```

## train

학습과 테스트시 main_monodepth_pytorch.py 를 사용하고 학습시 argument는 다음과 같다. :
 - `data_dir`: 학습 혹은 테스트 데이터 경로
 - `val_data_dir`:  validation 데이터 경로
 - `model_path`: 모델이 저장될 경로와 어떤 모델인지 이름으로 구분
 - `output_directory`: 테스트시 Depth 영상이 저장될 경로
 - `input_height` : 입력 영상 높이
 - `input_width` : 입력 영상 넒이
 - `model`:  encoder 모델 (resnet18_md or resnet50_md or any torchvision version of Resnet (resnet18, resnet34 etc.)
 - `pretrained`: Pretrained 된 resnet 모델을 사용할 경우 사용
 - `mode`: train or test
 - `epochs`: number of epochs,
 - `learning_rate` 
 - `batch_size` 
 - `adjust_lr`: Learning-late schedular를 사용할 것이지
 - `tensor_type`:'torch.cuda.FloatTensor' or 'torch.FloatTensor'
 - `do_augmentation`:do data augmentation or not
 - `augment_parameters`:lowest and highest values for gamma, lightness and color respectively
 - `print_images` : 학습시 영상을 저장하면서 할 것인지
 - `print_weights` : 학습시 모델을 출력할 것인지
 - `input_channels`: Number of channels in input tensor (3 for RGB images)
 - `num_workers`: Number of workers to use in dataloader
 - `RGB` : 모델의 입력으로 RGB를 사용할 것인지 아니면 열화상 영상을 사용할 것인지 

- 학습하기 위한 간단한 실행 명령어는 train.sh 에 있어서, 데이터 경로를 맞춰주고 train.sh을 실행 시켜주면 학습이 될 것이다.
```
bash train.sh
```
## test
테스트 argument는 학습과 동일하며 테스트 하기 위한 실행 명령어는 test.sh 에 있으니 그것을 실행 시키면 된다.

```
bash test.sh
```

### Requirements
This code was tested with PyTorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. Other required modules:

```
torchvision
numpy
matplotlib
```
# Result

## 정량적 평가

| model | domain| RMSE | RMSE_log |
|:-----: | :-----:|:-----: |:-----: |
| MTN    |   T    | 8.7387 | 0.1933 |
| Monodepth |   T  |  4.7079 |  0.1988 |
| Monodepth |   R  |  4.2886 |  0.2038  | 

## 정성적 평가


