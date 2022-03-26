import sys
import detect
from PyQt5 import QtWidgets
import cv2
import threading
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from GUI import Ui_Form
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



class MyPyQT_Form(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(MyPyQT_Form,self).__init__()
        self.resize(1200, 800)
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.vid_source = None
        self.busy=False
        self.stopEvent = threading.Event()
        self.webcam = False
        self.stopEvent.clear()
        self.model = self.model_load(weights="weights/best.pt",
                                     device=self.device)  # todo 指明模型加载的位置的设备
        self.setupUi(self)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.font = QFont()

        # 自定义卷积核--矩形，用于形态学处理
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # # 创建混合高斯模型用于背景建模
        self.back = cv2.createBackgroundSubtractorMOG2()

        # 大小
        self.font.setPointSize(10)
        self.font.setWeight(75)
        self.label.setFont(self.font)
        self.setWindowTitle('火灾图象智能识别系统')

    @torch.no_grad()
    def model_load(self, weights="'weights/best.pt'",  # model.pt path(s)
                   device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    #实现pushButton_click()函数
    def push_vid(self):
        self.vid_source, fileType = QFileDialog.getOpenFileName(self, '选择视频文件', '*.mp4')
        self.label.setText(self.vid_source)

    def begin_det(self):
        if self.busy and self.stopEvent.is_set() == False:
            self.font.setBold(False)
            self.label.setText('检测结束')
            self.overturn()
            self.stopEvent.set()
        else :
            if self.webcam:
                self.vid_source = '0'
            elif self.vid_source is None:
                QMessageBox.warning(self, '错误',"请选择视频来源再进行检测！")
                return
            self.detect_vid()



    def motion_det_MOG(self,frame):
        img = self.back.apply(frame)  # 背景建模
        # cv2.imshow('img', img)  # 高斯模型图
        # 开运算（先腐蚀后膨胀），去除噪声
        img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
        # 轮廓检测，获取最外层轮廓，只保留终点坐标
        contours, hierarchy = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算轮廓外接矩形
        motion_detections = []
        for cnt in contours:
            # 计算轮廓周长
            length = cv2.arcLength(cnt, True)
            if length > 100:
                # 得到外接矩形的要素
                x, y, w, h = cv2.boundingRect(cnt)
                # 画出这个矩形，在原视频帧图像上画，左上角坐标(x,y)，右下角坐标(x+w,y+h)
                # frame=cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                motion_detections.append((x, y, w, h))
        # cv2.imshow('frame', frame)
        return motion_detections
        # 图像展示
        #   # 原图
        # # 设置关闭条件，一帧200毫秒
        # k = cv2.waitKey(100) & 0xff
        # if k == 27:  # 27代表退出键ESC
        #     break

    def bb_overlab(self,b1, bm,im):
        '''
        说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
        :param x1: 第一个框的左上角 x 坐标
        :param y1: 第一个框的左上角 y 坐标
        :param w1: 第一幅图中的检测框的宽度
        :param h1: 第一幅图中的检测框的高度
        :param x2: 第二个框的左上角 x 坐标
        :param y2:
        :param w2:
        :param h2:
        :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
        '''
        try:
            ax1 = int(b1[0]-b1[2]/2)
            ay1 = int(b1[1]-b1[3]/2)
            ax2 = int(ax1+b1[2])
            ay2 = int(ay1+b1[3])
            for b2 in bm:
                bx1 = b2[0]
                by1 = b2[1]
                bx2 = bx1+b2[2]
                by2 = by1+b2[3]
                # 没有相交直接返回
                if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
                    continue
                # 交集面积
                intersection = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))
                # 并集面积 = 两个矩形面积相加 减去 交集面积
                and_set = b1[2] * b1[3] + b2[2] * b2[3] - intersection
                if intersection / and_set:
                    return True
        except:
            return False

    def detect_vid(self):
        self.overturn()
        # 加粗
        self.font.setBold(True)
        model = self.model
        output_size = self.output_size
        # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.3  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = 0  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 2  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt = [0.0, 0.0, 0.0]
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # Process predictions



            for i, det in enumerate(pred):  # per image
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # save_path = str(save_dir / p.name)  # im.jpg
                    # txt_path = str(save_dir / 'labels' / p.stem) + (
                    #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    pm = self.motion_det_MOG(im0)
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        fire=False
                        for *xyxy, conf, cls in reversed(det):

                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # xywh
                            if pm and self.bb_overlab(xywh, pm,im0):
                                fire=True
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if fire:self.label.setText("<font color=%s>%s</font>" %("#FF0000", "检测到明火！"))

                    # Print time (inference-only)
                    # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    # Save results (image with detections)
                    im0 = annotator.result()
                    frame = im0
                    resize_scale = output_size / frame.shape[0]
                    frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                    # cv2.imwrite("results/single_result_vid.jpg", frame_resized)
                    QtImgBuf = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
                    # 把转换格式后的帧图片，实例化QImage对象
                    QtImg = QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QImage.Format_RGB32)
                    # VideoPlayer是UI界面中的label对象。先用QtImg实例和QPixmap对象，然后将其传给label
                    self.frame_label.setPixmap(QPixmap.fromImage(QtImg))
                    # 使用QLabel的setScaledContents方法，是图片自适应QLabel的大小
                    self.frame_label.setScaledContents(True)
                    # self.vid_img
                    # if view_img:
                    # cv2.imshow(str(p), im0)
                    # self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                    # cv2.waitKey(1)  # 1 millisecond

                    if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                        self.stopEvent.clear()
                        return
            # Print time (Total duration of per image)
            t4=time_sync()
            LOGGER.info(f'{s}Done. ({t4 - t1:.3f}s)')

        self.font.setBold(False)
        self.label.setText('检测结束')
        self.frame_label.clear()
        self.overturn()
        QMessageBox.warning(self, '提示', "检测完成")

    def overturn(self):
        self.busy = not self.busy
        if self.busy:
            self.det.setText("停止")
        else:
            self.det.setText("开始检测")


    def from_camera(self):
        self.frame_label.clear()
        if self.webcam:
            self.webcam = False
            self.video_input.setEnabled(True)
            self.label.setText("空")
            self.label.setEnabled(True)
        else:
            self.webcam=True
            self.video_input.setEnabled(False)
            self.label.setText("（摄像头实时检测）")
            self.label.setEnabled(False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())