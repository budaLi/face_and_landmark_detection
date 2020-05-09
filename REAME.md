## Opencv 视频流人脸检测及关键点检测

### 安装 推荐使用清华镜像

     opencv

    pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

    pip install hdface

### 文件说明

    --face_and_landmark_detection  项目目录
       -checkpoint_epoch_120.pth    pytorch模型
       -flask_face_detection.py     主文件
       -pfld.py                     网络结构
       -req.txt                     所需环境

### run
    python flask_face_detection.py

### PS

    受hdface 人脸检测器影响 视频流比较卡顿 关键点定位还是比较准确的