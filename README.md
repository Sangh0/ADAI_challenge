# [ADAI 2022 challenge](https://adai2022.com/)  
- The Chanllenge of Developing a AI Algorithm for safe Autonomous Driving   
- My Assignment task: 2D Semantic Segmentation    

## Introduce of classes in dataset  
```
classes = {
    1: [255, 0, 0],      # non
    2: [128, 0, 0],      # road
    3: [255, 255, 0],    # full_line
    4: [128, 128, 0],    # dotted_line
    5: [0, 255, 0],      # road_mark
    6: [0, 128, 0],      # crosswalk
    7: [0, 255, 255],    # speed_bump
    8: [0, 128, 128],    # curb
    9: [0, 0, 255],      # static
    10: [0, 0, 128],     # sidewalk
    11: [255, 0, 255],   # parking_place
    12: [128, 0, 128],   # vehicle
    13: [255, 127, 80],  # motorcycle
    14: [184, 134, 11],  # bicycle
    15: [127, 255, 0],   # pedestrian
    16: [0, 191, 255],   # rider
    17: [255, 192, 203], # dynamic
    18: [165, 42, 42],   # traffic_sign
    19: [210, 105, 30],  # traffic_light
    20: [240, 230, 140], # pole
    21: [245, 245, 220], # building
    22: [0, 100, 0],     # guadrail
    23: [64, 224, 208],  # sky
    24: [70, 130, 180],  # water
    25: [106, 90, 205],  # mountain
    26: [75, 0, 130],    # vegetation
    27: [139, 0, 139],   # bridge
    28: [255, 20, 147],  # undefined/area
}
```  

## Model  
- [BiSeNetV2(2020)](https://arxiv.org/abs/2004.02147)  
- Review and Implementation of BiSeNetV2 in [my github](https://github.com/Sangh0/Segmentation/tree/main/BiSeNetV2)
- Reference: [Implementation Github](https://github.com/CoinCheung/BiSeNet)  
- The architecture of model  
<img src = "https://github.com/Sangh0/Segmentation/blob/main/BiSeNetV2/figure/figure3.JPG?raw=true">  

## Create environment on anaconda  
```
$ git clone this repository
$ cd ADAI_challenge
$ conda env create --file environment.yaml
$ conda activate adai2022
```

## Train
```
usage: main.py [-h] [--save_weight_dir SAVE_WEIGHT_DIR] [--weight_dir WEIGHT_DIR] [--data_dir DATA_DIR] \ 
	       [--data_mode DATA_MODE] [--lr LR] [--end_lr END_LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] \
	       [--weight_decay WEIGHT_DECAY] [--miou_weight MIOU_WEIGHT] [--celoss_weight CELOSS_WEIGHT] \
	       [--num_classes NUM_CLASSES] [--lr_scheduling LR_SCHEDULING] [--check_point CHECK_POINT] \
	       [--early_stop EARLY_STOP] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] \
	       [--train_log_step TRAIN_LOG_STEP] [--valid_log_step VALID_LOG_STEP]

example: 
$ python main.py --save_weight_dir ./current_date_weights --data_dir ./dataset/ --weight_dir ./weights/bisenetv2_pretrained.pth --num_classes 15
```

## Evaluate  
```
usage: eval.py [-h] [--weight_dir WEIGHT_DIR] [--data_dir DATA_DIR] [--num_classes NUM_CLASSES] \
               [--batch_size BATCH_SIZE] [--data_preprocess DATA_PREPROCESS]

example: 
$ python main.py --data_dir ./dataset --weight_dir ./weights/best_weight.pt --num_classes 15 --data_preprocess True
```

## Visualization in Jupyter Notebook  
```python
from eval import Evaluation

eval = Evaluation(
    path='./dataset',
    batch_size=8,
    num_classes=15,
    weight_path='./best_weight.pt',
)

result = eval.test()

eval.visualize(
    result['imags'],
    result['label'],
    result['output'],
    result['miou'],
    count=30,
)
```
