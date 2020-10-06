# character_detection
Faster RCNN , Yolo v3를 사용한 인물 탐지

# requirement

__0. 파이참 로딩__   
파이참 로딩 때 간혹 빈 프로젝트를 생성하고, 코드를 엎어씌우는 과정에서 오류가 나시는 분들이 있습니다.
프로젝트를 따로 만들지 않고, Pycharm > File > Open 탭으로 프로젝트를 열거나 터미널 상에서 접근하시면 큰 오류가 없을 것 같습니다.


__1. 필요 라이브러리 설치__   
( YOLO와 Faster RCNN이 서로 다른 Tensorflow버전 상에 동작하므로 
각기 다른 conda 환경을 만들어 동작할 것을 권고)
```
pip install -r requirements.txt
```
__2. 학습 데이터 다운로드__   

[다운로드 링크](https://drive.google.com/file/d/1lHP_92tj3pkonEoxwvqD_am7XsePbSl4/view?usp=sharing)   
--> 이미 LMS로부터 받으신 분은 해당 zip파일 그대로 사용하시면 됩니다. 혹시나 LMS에 전체 데이터가 다 올라가지 않을까 염려되어 링크를 올려둔 것이니, 참고하시기 바랍니다. 데이터.zip파일의 용량은 약 216MB이며, 압축 해제했을 경우엔 756MB 정도의 사이즈를 갖고 있습니다. 확인해보시고 문제 있으시면 다시 다운로드 받아 사용하세요.

- xml, img 리스트 파일 생성
```
 # Data폴더 내 python 파일 130 - 135라인에 대하여 본인 경로로 수정 후
 python make_file_list.py
 ```
test_img_filelist.txt / test_xml_filelist.txt / trainval_img_filelist.txt / trainval_xml_filelist.txt   
총 4개의 파일이 생성됐는지 확인

__3. 수정해야 하는 파일들__

__[Faster RCNN / YOLO 공통]__
- __경로 설정 시, Window는 "//" 를 Ubuntu에서는 "\\"를 유의해서 사용해주세요.__

- ./misaeng_annotation.py 9-13라인 수정
```
train_path="[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_filelist.txt"
imgsets_path_trainval = "[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_img"
imgsets_path_test = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_img"
test_path = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_filelist.txt"
```
- __간혹, 이미지를 로딩할때 (cv2.imread()) 등 NoneType에러가 나는 경우가 있는데 대다수 한글이 포함된 경로를 사용할 때 에러가 납니다. 코드 및 데이터 로딩 시 한글이 포함되지 않은 경로를 사용해주세요.__

- __각 모델 별 설정된 하이퍼 파라미터는 테스트 용으로 기입된 숫자이며, 절대 잘 되는 모델을 위한 하이퍼 파라미터가 아닙니다.본인 컴퓨터 사양에 맞춰 여러 번 실험해보고 최고의 성능을 내는 하이퍼 파라미터를 결정한 후, 실험에 사용하시기 바랍니다.__

## YOLO requirement
From (https://github.com/david8862/keras-YOLOv3-model-set)   

__1. YOLO v3 weight(컴퓨터 사양에 따라 YOLOv3-Tiny도 가능) 다운로드__
__2. Darknet YOLO 모델을 Keras모델로 변경__

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
  
__3. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정__

  ```
  python misaeng_annotation.py
  ```
  [filepath,x1,y1,x2,y2,class_id]형태로 두 가지 파일을 생성   
  : trainval_misaeng.txt / test_misaeng.txt
  
__4. 학습진행__

  ```
  python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --annotation_file=trainval_misaeng.txt --classes_path=configs/misaeng_classes.txt --eval_online --save_eval_checkpoint --val_annotation_file=test_misaeng.txt
  ```
__5. 성능 확인__   

  "--val_annotation_file" 옵션으로 테스트 데이터를 제공한 경우, 자동으로 학습이 끝난 후 네트워크 평가 진행
  
__6. Tensorboard 확인__   

   학습 과정 중 일어난 Loss변화 또는 학습 변화를 확인하고 싶을 때는 results/classes폴더 내의 이미지, html파일을 확인할수 있으며,
   logs/000/train(validation)폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/000/train(validation)
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음

  
## Faster RCNN
From (https://github.com/kbardool/keras-frcnn)

__1. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정(YOLO와 형식이 다르므로 한번 더 수행)__   

   ```
   python misaeng_annotation.py
   ```

   [filepath,x1,y1,x2,y2,class_name]형태로 두 가지 파일을 생성
   trainval_misaeng.txt / test_misaeng.txt 파일을 생성 

__2. 학습__   

   ```
   python train_frcnn.py -p trainval_misaeng.txt
   ```

__3. 평가__   

   ```
   python measure_map.py -o simple -p [TEST데이터]
   ex) python measure_map.py -o simple -p test_misaeng.txt
   ```
- __결과물이 나오지 않는 경우(모델이 인물 탐지를 전혀 하지 못하는 경우)는 충분히 학습 되지 않은 모델입니다.__   
- __rpn_cls 나 detector_cls 의 loss가 1 이상일 경우, Loss가 너무 높은 모델입니다. 조금 더 훈련을 진행하셔야 합니다.__
![error](https://user-images.githubusercontent.com/30281608/95056881-502ec880-0730-11eb-9f24-98f3dc163036.png)   

__다음과 같이 AP값이 1.0이 나올 경우, 모델이 제대로 된 탐지를 전혀 하지 못하는 경우를 의미합니다.__

__만약, mAP계산 외에 추가적으로 실제 모델의 탐지 결과를 확인하고 싶을 땐__
```
python test_frcnn.py -p [절대경로]/test_img
```
__다음 명령어를 통해 실제 Visualize한 탐지 결과(results_imgs 내 이미지)를 얻으실 수 있습니다.__

__4. Tensorboard 확인__   
   학습 과정 중 일어난 Loss변화를 확인하고 싶을 때는 logs/frcnn 폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/frcnn
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음
