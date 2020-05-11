# @Time    : 2020/5/11 10:58
# @Author  : Libuda
# @FileName: merge_image.py
# @Software: PyCharm

import os
import cv2
import time

# 图片合成视频
def picvideo(path, size):
    filelist = os.listdir(path)  # 获取该目录下的所有文件名

    # 避免跳帧
    filelist.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 12
    # size = (591,705) #图片的分辨率片
    file_path = r"./vedio/" + str(int(time.time())) + ".mp4"  # 导出路径
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.jpg'):  # 判断图片后缀
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频

    video.release()  # 释放


# 图片大小 640*480
picvideo(r'./image', (640, 480))

print("end")