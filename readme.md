# character_detection
Faster RCNN , Yolo v3를 사용한 인물 탐지

# requirement

1. 필요 라이브러리 설치   
( YOLO와 Faster RCNN이 서로 다른 Tensorflow버전 상에 동작하므로 각기 다른 conda 환경을 만들어 동작할 것을 권고)
```
pip install -r requirements.txt
```
2. 학습 데이터 다운로드   

[추후 수정]

- xml, img 리스트 파일 생성
```
 # Data폴더 내 python 파일 130 - 135라인에 대하여 본인 경로로 수정 후
 python make_file_list.py
 
```
test_img_filelist.txt / test_xml_filelist.txt
trainval_img_filelist.txt / trainval_xml_filelist.txt 
총 4개의 파일이 생성됐는지 확인

3. 수정해야 하는 경로 파일들   

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
  trainval_misaeng.txt / test_misaeng.txt 파일을 생성 
  
4. 학습진행
  ```
  python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --annotation_file=trainval_misaeng.txt --classes_path=configs/misaeng_classes.txt --eval_online --save_eval_checkpoint --val_annotation_file=test_misaeng.txt
  ```
5. 성능 확인
  "--val_annotation_file" 옵션으로 테스트 데이터를 제공하였으므로, 학습이 끝난 후 test 셋을 바탕으로 네트워크 평가 진행
  
6. Tensorboard 확인   
   학습 과정 중 일어난 Loss변화 또는 학습 변화를 확인하고 싶을 때는 results/classes폴더 내의 이미지, html파일을 확인할수 있으며,
   logs/000/train(validation)폴더 안에 있는 tensorboard 파일로 변화율을 확인할 수 있음
   ```
   tensorboard --logdir=logs/000/train(validation)
   ```
   
   [환경 라이브러리 정보]
   ```
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
absl-py                   0.10.0                    <pip>
aiohttp                   3.6.2                     <pip>
astor                     0.8.1                     <pip>
astunparse                1.6.3                     <pip>
async-timeout             3.0.1                     <pip>
attrs                     20.2.0                    <pip>
bokeh                     2.2.1                     <pip>
ca-certificates           2020.7.22                     0  
cachetools                4.1.1                     <pip>
certifi                   2020.6.20                py36_0  
chardet                   3.0.4                     <pip>
coremltools               3.4                       <pip>
cycler                    0.10.0                    <pip>
Cython                    0.29.21                   <pip>
decorator                 4.4.2                     <pip>
dm-tree                   0.1.5                     <pip>
fire                      0.3.1                     <pip>
flatbuffers               1.12                      <pip>
gast                      0.3.3                     <pip>
google-auth               1.22.0                    <pip>
google-auth-oauthlib      0.4.1                     <pip>
google-pasta              0.2.0                     <pip>
graphviz                  0.14.1                    <pip>
grpcio                    1.32.0                    <pip>
h5py                      2.10.0                    <pip>
idna                      2.10                      <pip>
idna-ssl                  1.1.0                     <pip>
imagecorruptions          1.1.0                     <pip>
imageio                   2.9.0                     <pip>
imgaug                    0.4.0                     <pip>
importlib-metadata        2.0.0                     <pip>
Jinja2                    2.11.2                    <pip>
Keras-Applications        1.0.8                     <pip>
Keras-Preprocessing       1.1.2                     <pip>
keras2onnx                1.7.0                     <pip>
kiwisolver                1.2.0                     <pip>
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
Markdown                  3.2.2                     <pip>
MarkupSafe                1.1.1                     <pip>
matplotlib                3.3.2                     <pip>
MNN                       1.0.3                     <pip>
mpmath                    1.1.0                     <pip>
multidict                 4.7.6                     <pip>
ncurses                   6.2                  he6710b0_1  
networkx                  2.5                       <pip>
numpy                     1.17.5                    <pip>
oauthlib                  3.1.0                     <pip>
onnx                      1.7.0                     <pip>
onnxconverter-common      1.7.0                     <pip>
onnxruntime               1.5.1                     <pip>
opencv-contrib-python     4.4.0.44                  <pip>
opencv-python             4.4.0.44                  <pip>
openssl                   1.1.1h               h7b6447c_0  
opt-einsum                3.3.0                     <pip>
packaging                 20.4                      <pip>
Pillow                    7.2.0                     <pip>
pip                       20.2.3                   py36_0  
protobuf                  3.13.0                    <pip>
pyasn1                    0.4.8                     <pip>
pyasn1-modules            0.2.8                     <pip>
pycocotools               2.0.2                     <pip>
pydot-ng                  2.0.0                     <pip>
pyparsing                 2.4.7                     <pip>
python                    3.6.12               hcff3b4d_2  
python-dateutil           2.8.1                     <pip>
PyWavelets                1.1.1                     <pip>
PyYAML                    5.3.1                     <pip>
readline                  8.0                  h7b6447c_0  
requests                  2.24.0                    <pip>
requests-oauthlib         1.3.0                     <pip>
rsa                       4.6                       <pip>
scikit-image              0.17.2                    <pip>
scipy                     1.4.1                     <pip>
setuptools                49.6.0                   py36_1  
Shapely                   1.7.1                     <pip>
six                       1.15.0                    <pip>
sqlite                    3.33.0               h62c20be_0  
sympy                     1.6.2                     <pip>
tensorboard               2.3.0                     <pip>
tensorboard-plugin-wit    1.7.0                     <pip>
tensorflow                1.14.0                    <pip>
tensorflow-estimator      2.3.0                     <pip>
tensorflow-gpu            2.3.0                     <pip>
tensorflow-model-optimization 0.5.0                     <pip>
termcolor                 1.1.0                     <pip>
tf2onnx                   1.6.3                     <pip>
tfcoreml                  2.0                       <pip>
tifffile                  2020.9.3                  <pip>
tk                        8.6.10               hbc83047_0  
tornado                   6.0.4                     <pip>
tqdm                      4.50.0                    <pip>
typing-extensions         3.7.4.3                   <pip>
urllib3                   1.25.10                   <pip>
Werkzeug                  1.0.1                     <pip>
wheel                     0.35.1                     py_0  
wrapt                     1.12.1                    <pip>
xz                        5.2.5                h7b6447c_0  
yarl                      1.6.0                     <pip>
zipp                      3.2.0                     <pip>
zlib                      1.2.11               h7b6447c_3

   ```

  
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

   라이브러리 정보
   ```
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
absl-py                   0.10.0                    <pip>
astor                     0.8.1                     <pip>
ca-certificates           2020.7.22                     0  
certifi                   2020.6.20                py36_0  
gast                      0.4.0                     <pip>
grpcio                    1.32.0                    <pip>
h5py                      2.10.0                    <pip>
importlib-metadata        2.0.0                     <pip>
joblib                    0.16.0                    <pip>
Keras                     2.0.3                     <pip>
Keras-Applications        1.0.8                     <pip>
Keras-Preprocessing       1.1.2                     <pip>
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
Markdown                  3.2.2                     <pip>
ncurses                   6.2                  he6710b0_1  
numpy                     1.19.2                    <pip>
opencv-python             4.4.0.44                  <pip>
openssl                   1.1.1h               h7b6447c_0  
pip                       20.2.3                   py36_0  
protobuf                  3.13.0                    <pip>
python                    3.6.12               hcff3b4d_2  
PyYAML                    5.3.1                     <pip>
readline                  8.0                  h7b6447c_0  
scikit-learn              0.23.2                    <pip>
scipy                     1.5.2                     <pip>
setuptools                49.6.0                   py36_1  
six                       1.15.0                    <pip>
sklearn                   0.0                       <pip>
sqlite                    3.33.0               h62c20be_0  
tensorboard               1.12.2                    <pip>
tensorflow                1.12.0                    <pip>
termcolor                 1.1.0                     <pip>
Theano                    1.0.5                     <pip>
threadpoolctl             2.1.0                     <pip>
tk                        8.6.10               hbc83047_0  
Werkzeug                  1.0.1                     <pip>
wheel                     0.35.1                     py_0  
xz                        5.2.5                h7b6447c_0  
zipp                      3.2.0                     <pip>
zlib                      1.2.11               h7b6447c_3 

   ```