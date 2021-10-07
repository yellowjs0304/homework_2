# Character_detection
Faster RCNN , Yolo v3를 사용한 인물 탐지   
__수정된 코드가 있으므로, LMS에서 다운받은 코드가 아닌 GitHub에서 코드를 재 다운로드 하여 사용해주세요__   
__지속적으로 GitHub readme를 업데이트 하고 있습니다. 오류가 생기면 우선 Readme를 한번 읽어보시고, 질문 주시면 감사하겠습니다.__

### 자주 질문하는 오류(이외에도 아래 readme 전체를 자세히 읽으시면 관련 내용이 있습니다)   

- simple_parser.py", line 37, in get_data (rows,cols) = img.shape[:2] AttributeError: 'NoneType' object has no attribute 'shape'   
: 한글 경로, misaeng_annotation.py에서 .split함수 미수정, 경로 오류(\\\\, / 등)   
: Ubuntu를 사용하시는 분들은 자동 공백제거가 안되서, Image를 못 읽으시는 경우가 있습니다. FRCNN안에 misaeng_annotation.py에서 상단 코드의 공백을 아래와 같이 삭제해주시면 해당 오류를 삭제하실 수 있습니다.
```
기존 : list_file.write(os.path.join(imgsets_path, line.split('/')[-1][:-4]+"jpg ,")) 
변경 : list_file.write(os.path.join(imgsets_path, line.split('/')[-1][:-4]+"jpg,"))
```

- FileNotFoundError: [Errono 2] No such file or directory: 'C:\\Users\\JSHwang\\...\\...jpg'   
: misaeng_annotation.py의 결과로 생성된 trainval_misaeng, test_misaeng파일이 Yolo, Keras 폴더 바로 아래에 존재해야 함. 즉, train.py 코드와 동일한 위치   

- GPU를 쓸 때는 각자의 Nvidia 그래픽 드라이버에 맞춰서 CUDA와 CuDNN을 설치해주셔야 하며, CUDA 10.1, cuDNN은 7.4 ~ 7.6 대를 쓰시면 됩니다.   

- 환경 셋팅 시, 여러 개의 tensorflow가 설치되어 있는 분들이 있습니다. GPU가 있는 분들은 -gpu버전을 설치해주셔도 무방하지만, 없으신 분들은 requirement에서 "-gpu"를 -cpu로 변경하고 설치해주셔야 돼요. 밑에까지 안읽고 그냥 설치하시는 분들이 많아 한번 더 명시해드립니다.

# Requirement

### __0. 파이참 로딩__ ###   
파이참 로딩 때 간혹 빈 프로젝트를 생성하고, 코드를 엎어씌우는 과정에서 오류가 나시는 분들이 있습니다.
프로젝트를 따로 만들지 않고, Pycharm > File > Open 탭으로 프로젝트를 열거나 터미널 상에서 접근하시면 큰 오류가 없을 것 같습니다.


### __1. 필요 라이브러리 설치__ ###   
( YOLO와 Faster RCNN이 서로 다른 Tensorflow버전 상에 동작하므로 
각기 다른 conda 환경을 만들어 동작할 것을 권고)
```
pip install -r requirements.txt
```
- 설치 과정 중 coviar, tfcoreml, pycocotools 관련 오류가 발생하시는 분들은 GitHub에서 requirements파일을 다시 다운받거나 해당 라인을 지우고 재설치 해보시기 바랍니다.   
- tfcoreml은 tensorflow버전에 맞춰서 설치하셔도 됩니다.   
- python은 두 모델 모두 3.6버전을 사용하시기 바랍니다.   
- YOLO는 텐서플로 2.3.0을 Frcnn은 1.12.0(또는 1.14.0)을 사용하면 되며, GPU 유무에 따라 requirements 파일에서 "-gpu"를 삭제, 추가하실 수 있습니다.   

### __2. 학습 데이터 다운로드__ ###   

~~[다운로드 링크]()~~   

--> 이미 LMS로부터 받으신 분은 해당 zip파일 그대로 사용하시면 됩니다. 혹시나 LMS에 전체 데이터가 다 올라가지 않을까 염려되어 링크를 올려둔 것이니, 참고하시기 바랍니다. 데이터.zip파일의 용량은 약 216MB이며 확인해보시고 문제 있으시면 담당조교에게 문의 바랍니다.   

- xml, img 리스트 파일 생성
```
 # Data폴더 내 python 파일 130 - 135라인에 대하여 본인 경로로 수정 후
 python make_file_list.py
 ```
test_img_filelist.txt / test_xml_filelist.txt / trainval_img_filelist.txt / trainval_xml_filelist.txt   
총 4개의 파일이 생성됐는지 확인

### __3. 수정해야 하는 파일들__ ###

__[Faster RCNN / YOLO 공통]__
- misaeng_annotation.py 9-13라인 수정   
(YOLO의 경우, tools/dataset_converter폴더 내에 있습니다)
```
train_path="[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_xml_filelist.txt"
imgsets_path_trainval = "[본인 경로에 맞도록 수정(절대경로)]\\data\\trainval_img"
imgsets_path_test = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_img"
test_path = "[본인 경로에 맞도록 수정(절대경로)]\\data\\test_xml_filelist.txt"
```
- 경로 설정 시, Window는 "\\\\" 를 Ubuntu에서는 "/"를 유의해서 사용해주세요.
- 이때, Windows와 Ubuntu에 따라서 misaeng_annotation.py파일의 46-47번째 라인 .split('\\\\') 구문과 .split('/')구문을 같이 변경해주셔야 합니다.
- 간혹, 이미지를 로딩할때 (cv2.imread()) 등 NoneType에러가 나는 경우가 있는데 대다수 한글이 포함된 경로를 사용할 때 에러가 납니다. 코드 및 데이터 로딩 시 한글이 포함되지 않은 경로를 사용해주세요.
- 각 모델 별 설정된 하이퍼 파라미터는 테스트 용으로 기입된 숫자이며, 절대 잘 되는 모델을 위한 하이퍼 파라미터가 아닙니다.본인 컴퓨터 사양에 맞춰 여러 번 실험해보고 최고의 성능을 내는 하이퍼 파라미터를 결정한 후, 실험에 사용하시기 바랍니다.   
- 코드를 실행할 때 너무 오래걸리거나 CPU, GPU용량이 부족할 경우 YOLO는 batch size를 줄이거나 Tiny-yolo를 고, FasterRCNN은 RoI의 수를 낮추거나 epoch_lengths, num_epoch등을 줄여 돌리시기 바랍니다.   
- 3번 과제를 위해 원래는 두 모델 다 동일한 CNN을 사용하는 것이 옳으나, 베이스라인 코드의 문제로 각기 다른 CNN으로 수행하셔도 됩니다.   
추후, 문제가 해결되면 Readme를 업데이트하겠습니다.
------------

## YOLO requirement
From (https://github.com/david8862/keras-YOLOv3-model-set)   

### __1. YOLO v3 weight(컴퓨터 사양에 따라 YOLOv3-Tiny도 가능) 다운로드__ ###
### __2. Darknet YOLO 모델을 Keras모델로 변경__ ###   

- 다운 받은 weight파일을 weights폴더에 복사   

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
  
### __3. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정__ ###

  ```
  python misaeng_annotation.py
  ```
  [filepath x1,y1,x2,y2,class_id]형태로 두 가지 파일을 생성   
  : trainval_misaeng.txt / test_misaeng.txt
  
### __4. 학습진행__ ###

  ```
  # YOLO_ANCHOR_TXT는 YOLO모델에 따라 다름 = Tiny YOLO의 Anchor개수 : 6 / YOLOv3 개수 : 9개
  python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/[YOLO_ANCHOR_TXT] --annotation_file=[TRAIN_DATA_FILE] --classes_path=configs/[CLASS_FILE_TXT] --eval_online --save_eval_checkpoint --val_annotation_file=[TEST_DATA_FILE]
  ex) python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --annotation_file=trainval_misaeng.txt --classes_path=configs/misaeng_classes.txt --eval_online --save_eval_checkpoint --val_annotation_file=test_misaeng.txt
  
  TINY Model의 경우, 명령어 예제
  ex) python train.py --model_type=tiny_yolo3_mobilenet_lite --anchors_path=configs/tiny_yolo3_anchors.txt --annotation_file=trainval_misaeng.txt --classes_path=configs/misaeng_classes.txt --eval_online --save_eval_checkpoint --val_annotation_file=test_misaeng.txt
  ```
### __5. 성능 확인__ ###   

  "--val_annotation_file" 옵션으로 테스트 데이터를 제공한 경우, 자동으로 학습이 끝난 후 네트워크 평가 진행
  
  __만약, mAP계산 외에 추가적으로 실제 모델의 탐지 결과를 확인하고 싶을 땐__
```
python yolo.py --model_type=yolo3_mobilenet_lite --weights_path=[MODEL_NAME] --anchors_path=configs/[ANCHOR_FILE] --classes_path=configs/misaeng_classes.txt --model_image_size=416x416 --image

ex) python yolo.py --model_type=yolo3_mobilenet_lite --weights_path=model.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/voc_classes.txt --model_image_size=416x416 --image

이후, cmd창에 image path를 입력
```
__다음 명령어를 통해 실제 Visualize한 탐지 결과(results_imgs 내 이미지)를 얻으실 수 있습니다.__

### __6. Tensorboard 확인__ ###   

   학습 과정 중 일어난 Loss변화 또는 학습 변화를 확인하고 싶을 때는 results/classes폴더 내의 이미지, html파일을 확인할수 있으며,
   logs/000/train(validation)폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/000/train(validation)
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음


------------

## Faster RCNN
From (https://github.com/kbardool/keras-frcnn)

### __1. Annotation 실행하여 본인 경로에 맞는 데이터 셋 설정(YOLO와 형식이 다르므로 한번 더 수행)__ ###   

   ```
   python misaeng_annotation.py
   ```

   [image_file_path box1 box2 ... boxN] (box N : x_min,y_min,x_max,y_max,class_id)형태로 두 가지 파일을 생성   
   trainval_misaeng.txt / test_misaeng.txt 파일을 생성   
   
   - __만약 Class name이 아니라 0-4에 해당되는 숫자값으로 생성됐을 경우, GitHub에 있는 misaeng_annotation.py 파일로 변경해보시고 다시 사용해보시기 바랍니다. LMS에 올렸던 코드에서 한번 업데이트가 있던 코드라 차이가 있을 수 있습니다.__

### __2. 학습__ ###   

   ```
   python train_frcnn.py -p trainval_misaeng.txt
   ```
   - FRCNN의 batch size는 epoch length에 의거하여 변경됩니다. 즉, epoch length를 설정하는 것이 batch size를 결정하는 것과 동일한 역할을 합니다.   
   - epoch length = "전체 train data set 개수 // batch size " 입니다.   
   ex) 전체 train 개수 3000개라 가정했을 때, batch size가 10일 경우 epoch length는 300으로 설정하시면 됩니다.   
   - misaeng 학습 데이터 개수는 train_frcnn.py를 실행했을 때 cmd창에 명시되는 Num train samples 값을 사용하시면 되겠습니다.   
   
### __3. 평가__ ###   

   ```
   python measure_map.py -o simple -p [TEST데이터]
   ex) python measure_map.py -o simple -p test_misaeng.txt
   ```
- __결과물이 나오지 않는 경우(모델이 인물 탐지를 전혀 하지 못하는 경우)는 충분히 학습 되지 않은 모델입니다.__   
- __rpn_cls 나 detector_cls 의 loss가 1 이상일 경우, Loss가 너무 높은 모델입니다. 조금 더 훈련을 진행하셔야 합니다.__   
![error](https://user-images.githubusercontent.com/30281608/95056881-502ec880-0730-11eb-9f24-98f3dc163036.png)   

__다음과 같이 AP값이 1.0이 나오면서 반복될 경우, 모델이 제대로 된 탐지를 전혀 하지 못하는 경우를 의미합니다.__

__만약, mAP계산 외에 추가적으로 실제 모델의 탐지 결과를 확인하고 싶을 땐__
```
python test_frcnn.py -p [절대경로]/test_img
```
__다음 명령어를 통해 실제 Visualize한 탐지 결과(results_imgs 내 이미지)를 얻으실 수 있습니다.__

### __4. Tensorboard 확인__ ###   

학습 과정 중 일어난 Loss변화를 확인하고 싶을 때는 logs/frcnn 폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/frcnn
   ```
   
   [환경 라이브러리 정보] : library.txt파일에 기입되어 있음
