# [ADAI 2022 challenge](https://adai2022.com/)
- 안전한 자율주행을 위한 인공지능 알고리즘 개발 챌린지  
- task: 2D semantic segmentation   

## Introduce of classes in dataset  
```
classes = {
    1: [255, 0, 0], # non
    2: [128, 0, 0], # road
    3: [255, 255, 0], # full_line
    4: [128, 128, 0], # dotted_line
    5: [0, 255, 0], # road_mark
    6: [0, 128, 0], # crosswalk
    7: [0, 255, 255], # speed_bump
    8: [0, 128, 128], # curb
    9: [0, 0, 255], # static
    10: [0, 0, 128], # sidewalk
    11: [255, 0, 255], # parking_place
    12: [128, 0, 128], # vehicle
    13: [255, 127, 80], # motorcycle
    14: [184, 134, 11], # bicycle
    15: [127, 255, 0], # pedestrian
    16: [0, 191, 255], # rider
    17: [255, 192, 203], # dynamic
    18: [165, 42, 42], # traffic_sign
    19: [210, 105, 30], # traffic_light
    20: [240, 230, 140], # pole
    21: [245, 245, 220], # building
    22: [0, 100, 0], # guadrail
    23: [64, 224, 208], # sky
    24: [70, 130, 180], # water
    25: [106, 90, 205], # mountain
    26: [75, 0, 130], # vegetation
    27: [139, 0, 139], # bridge
    28: [255, 20, 147], # undefined/area
}
```  

## Model  
- [BiSeNetV2(2020)](https://arxiv.org/abs/2004.02147)  
- [My GitHub](https://github.com/Sangh0/Segmentation/tree/main/BiSeNetV2) Review and Implementation of BiSeNetV2 in github   
- [Reference](https://github.com/CoinCheung/BiSeNet)  
- The architecture of model  
<img src = "https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/figure/figure3.JPG?raw=true">  

## Create environment on anaconda  
```
conda create -n adai2022 python=3.8
conda activate adai2022
pip install -r requirements.txt
```

## Train
```
usage: main.py [-h] [--weight_dir WEIGHT_DIR] [--data_dir DATA_DIR] [--lr LR] [--epochs EPOCHS] \
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY] \[--num_classes NUM_CLASSES] \
               [--lr_scheduling LR_SCHEDULING] [--check_point CHECK_POINT] [--early_stop EARLY_STOP] \
               [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--train_log_step TRAIN_LOG_STEP] \
               [--valid_log_step VALID_LOG_STEP]

example: python main.py --data_dir ./dataset --weight_dir ./weights/bisenetv2_city.pth --num_classes 28
```

## Evaluate  
- Not Implemenetation