# character_detection
Faster RCNN , Yolo v3를 사용한 인물 탐지

# requirement

__1. 필요 라이브러리 설치__   
( YOLO와 Faster RCNN이 서로 다른 Tensorflow버전 상에 동작하므로 
각기 다른 conda 환경을 만들어 동작할 것을 권고)
```
pip install -r requirements.txt
```
__2. 학습 데이터 다운로드__  

[추후 수정]

- xml, img 리스트 파일 생성
```
 # Data폴더 내 python 파일 130 - 135라인에 대하여 본인 경로로 수정 후
 python make_file_list.py
 ```
test_img_filelist.txt / test_xml_filelist.txt/trainval_img_filelist.txt / trainval_xml_filelist.txt   
총 4개의 파일이 생성됐는지 확인

__3. 수정해야 하는 경로 파일들__

__[Faster RCNN / YOLO 공통]__

- ./misaeng_annotation.py 9-13라인 수정
```
train_path="[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_filelist.txt"
imgsets_path_trainval = "[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_img"
imgsets_path_test = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_img"
test_path = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_filelist.txt"
```

## YOLO requirement
From (https://github.com/david8862/keras-YOLOv3-model-set)
1. YOLO v3 weight(컴퓨터 사양에 따라 YOLOv3-Tiny도 가능) 다운로드
2. Darknet YOLO 모델을 Keras모델로 변경

  ```
  //Ubuntu   
  wget https://pjreddie.com/media/files/yolov3.weights   
  wget -O weights/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights   
  
  //windows   
  https://pjreddie.com/media/files/yolov3.weights로 이동   
  https://pjreddie.com/media/files/yolov3-tiny.weights    
  
  
  //YOLO v3   
  python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5   
  
  //YOLO v3 Tiny   
  python tools/model_converter/convert.py cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.h5   
  
  ```
  
3. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정
  ```
  python misaeng_annotation.py
  ```
  [filepath,x1,y1,x2,y2,class_id]형태로 두 가지 파일을 생성   
  : trainval_misaeng.txt / test_misaeng.txt
  
4. 학습진행
  ```
  python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --annotation_file=trainval_misaeng.txt --classes_path=configs/misaeng_classes.txt --eval_online --save_eval_checkpoint --val_annotation_file=test_misaeng.txt
  ```
5. 성능 확인   
  "--val_annotation_file" 옵션으로 테스트 데이터를 제공한 경우, 자동으로 학습이 끝난 후 네트워크 평가 진행
  
6. Tensorboard 확인   
   학습 과정 중 일어난 Loss변화 또는 학습 변화를 확인하고 싶을 때는 results/classes폴더 내의 이미지, html파일을 확인할수 있으며,
   logs/000/train(validation)폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/000/train(validation)
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음

  
## Faster RCNN
From (https://github.com/kbardool/keras-frcnn)

1. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정(YOLO와 형식이 다르므로 한번 더 수행)   

   ```
   python misaeng_annotation.py
   ```

   [filepath,x1,y1,x2,y2,class_name]형태로 두 가지 파일을 생성
   trainval_misaeng.txt / test_misaeng.txt 파일을 생성 

2. 학습   

   ```
   python train_frcnn.py -p trainval_misaeng.txt

   ```

3. 평가   

   ```
   python measure_map.py -o simple -p [TEST데이터]
   ex) python measure_map.py -o simple -p test_misaeng.txt
   ```

4. Tensorboard 확인   
   학습 과정 중 일어난 Loss변화를 확인하고 싶을 때는 logs/frcnn 폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/frcnn
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음
