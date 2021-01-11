# Pseudo-RGB

## Dataloader

데이터는 아래 구조와 같이 구성되어야만 합니다.

```
datas
├── train
│   ├─ RGB
│   │   ├── LEFT_000000000.jpg
│   │   └── ...
│   ├─ THER
│   │   ├── THER_000000000.jpg
│   │   └── ...
├── test
│   ├─ RGB
│   │   ├── LEFT_000000000.jpg
│   │   └── ...
│   ├─ THER
│   │   ├── THER_000000000.jpg
│   │   └── ...
```

이때 RGB의 영상명이 LEFT로 되어있는데, 이는 KAIST DATASET에서 제공되는 left RGB을 사용하기 때문이므로, 파일이름은 크게 신경 쓸 필요가 없다.

만약 Pseudo-Lidar와 동일한 데이터 폴더 구조를 가지고 있다면 아래와 같이 sum_image.py를 실행함으로써 위의 구조로 변경가능하다. 
```
python sum_image.py --train
```
--train argument를 추가 시 train data만을 불러오며, train argmuent를 제외하면 test data를 불러온다.


## train

1. 학습에는 train.py 파일을 사용하며 이때 사용되는 주요 argument는 다음과 같다. :
 - `stage`: 해당 모델은 총 3개의 network로 구성되어 있으며 학습시킬 네트워크 구간을 의미함.
 - `train_color_img_dir`: 학습할 RGB 영상의 폴더 경로
 - `train_thermal_img_dir`: 학습할 Thermal 영상의 폴더 경로
 - `name`: 실험결과가 저장되는 폴더 이름, 모델의 체크포인트가 해당 폴더에 저장됨.
 - `input_ther` : Thermal 영상을 학습시키는지, grey 영상을 학습시키는지 설정.

 그 외에 다른 argument들은 ./options/train_options.py에서 확인 가능하다.

2. scripts/prepare_train_box.sh'sL1과 scripts/train.sh's L1에 데이터 경로를 알맞게 맞춘 후 Instance Prediction을 먼저 실행한다.
```
sh scripts/prepare_train_box.sh
```
검출된 Instance box는 $DATASET_DIR 폴더에 저장된다. 

3. 전체 학습 과정을 간단하게 실행시키기 위한 명령어는 아래와 같다.
```
sh scripts/train.sh
```
학습 과정은 3단계로 진행된다.
a. 전체 영상에 대한 Colorization network(Full network)를 학습한다.
b. a에서 학습한 Full network의 checkpoint를 사용하여 instance Colorization network(Instance network)를 학습한다.
c. 마지막으로 Full network와 Instance network를 fusion한 fusion network를 학습한다.

## test
학습된 체크포인트들은 checkpoints/mask 폴더 내에 존재한다. 만약 학습하지 않고 미리 제공된 체크포인트로 평가하려면 [해당 드라이브](https://drive.google.com/drive/folders/1334v01UOgCG1A8wrDlgaXNcjuM-zjj85?usp=sharing)에서 체크포인트를 다운받아 checkpoints/mask 폴더 내에 저장한 후 평가한다.

아래 명령어를 실행하면 학습된 모델을 이용하여 colorization 된 영상이 생성되며 이는 $DATASET_DIR에 저장된다.
```
bash test.sh
```
정량적 결과를 확인하기 위해서 eval.py 내 predict 폴더 경로를 맞춘 후 아래 명령어를 실행한다.
```
python eval.py
```
##Demo
test_mask.sh 파일에 INPUT_DIR을 example로 변경 후 아래 명령어를 실행하면 example image에 대한 Colorization을 사용가능하다.
```
bash test.sh
```
## 정량적 결과

- KAIST2017 Dataset

|  | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:|:-----: |:-----: |
| Gray2RGB    |   35.0415    | 0.9692 | 0.0822 |
| Ther2RGB |   27.9761  |  0.4052 |  0.5074 |

- R2T2 Dataset

|  | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:|:-----: |:-----: |
| Gray2RGB    |   34.4895    | 0.9519 | 0.0822 |
| Ther2RGB |  27.9214  |  0.4422 |  0.5276 |

## 정성적 결과

![그림1.png](image/그림1.png)
