# @Time    : 2020/5/9 14:48
# @Author  : Libuda
# @FileName: flask_face_detection.py
# @Software: PyCharm
from hdface.hdface import hdface_detector
import cv2
import os
import shutil
from torchvision import transforms
# from pfld import PFLDInference
from pfld_simple import PFLDInference
import torch
import numpy as np

def detect_face(image):
    """
    从图片二维矩阵中检测出人脸
    :param image:
    :return:
    """
    det = hdface_detector(use_cuda=False)
    # img_det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = det.detect_face(image)
    boxs = []
    for one in result:
        face = one['box']
        boxs.append(face)

    return boxs

def op_vedio():
    # 多个摄像头 索引
    capture = cv2.VideoCapture(0)

    while 1:
        # frame 图片的每一帧
        ret, image = capture.read()

        # 是否成功读取摄像头
        if not ret:
            break

        # 镜像调换 上下 1 为正
        image = cv2.flip(image, 1)

        # face_boxs = detect_face(image)
        #
        # print(face_boxs)

        cv2.imshow("video", image)


        c = cv2.waitKey(20)
        # esc退出
        if c == 27:
            break


def detect_landmark(input):
    plfd_backbone = PFLDInference()
    angle, landmarks = plfd_backbone(input)

    print(landmarks)


def main():

    # 初始化网络
    plfd_backbone = PFLDInference()

    # 加载模型
    checkpoint = torch.load("checkpoint_epoch_120.pth",map_location='cpu')
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone
    transform = transforms.Compose([transforms.ToTensor()])

    # 读取摄像头图片
    # 多个摄像头 索引
    capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    image_path = "./image"
    if os.path.isfile(image_path) and len(os.listdir(image_path))>0:
        shutil.rmtree(image_path)
    print("摄像头加载完成")

    if not os.path.exists(image_path):
        os.mkdir(image_path)
    for i in range(1000):
        # frame 图片的每一帧
        ret, image = capture.read()
        # 是否成功读取摄像头
        if not ret:
            break
        # 镜像调换 上下 1 为正
        image = cv2.flip(image, 1)
        face_boxs = detect_face(image)
        # print(face_boxs)
        # face_boxs = [(250, 196, 400, 391)]
        height, width = image.shape[:2]

        if not face_boxs:
            continue
        box = face_boxs[0]
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 25))
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size_w = int(max([w, h]) * 0.9)
        size_h = int(max([w, h]) * 0.9)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size_w // 2
        x2 = x1 + size_w
        y1 = cy - int(size_h * 0.4)
        y2 = y1 + size_h

        left = 0
        top = 0
        bottom = 0
        right = 0
        if x1 < 0:
            left = -x1
        if y1 < 0:
            top = -y1
        if x2 >= width:
            right = x2 - width
        if y2 >= height:
            bottom = y2 - height

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = image[y1:y2, x1:x2]
        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0)
        pose, landmarks = plfd_backbone(input)

        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]


        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0))
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(image, (x1 - left + x, y1 - bottom + y), 1, (255, 255, 0), 1)

        cv2.imwrite("./image/test_{}.jpg".format(i),image)
        cv2.imshow("video", image)

        # 进行操作进行下一帧
        if cv2.waitKey():
            continue

        c = cv2.waitKey(10)
        # esc退出

if __name__ == '__main__':
    # op_vedio()
    main()