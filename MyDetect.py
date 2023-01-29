import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import classifyTrash

'''
conf_thres: 
iou_thres:
name: 保存的文件夹
project: 想要保存的目录
source：希望测试的图片
weights: 训练以后的权重文件
'''
class Detect:
    def __init__(self, conf_thres=0.25, iou_thres=0.45, name='exp',
                 project='runs/detect', source='testfiles/testfiles/img8.jpg',
                 weights='weights/best_300.pt'):
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        self.conf_thres = conf_thres
        self.device = ''
        self.exist_ok = False
        self.img_size = 640
        self.iou_thres = iou_thres
        self.name = name
        self.nosave = False
        self.project = project
        self.save_conf = False
        self.save_txt = False
        self.source = source
        self.update = False
        self.view_img = False
        self.weights = weights
        self.output=[]

    def detect(self):
        save_img = not self.nosave and not self.source.endswith('.txt')
        # 保存目录
        save_dir = Path(Path(self.project) / self.name)
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'
        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if half:
            model.half()  # to FP16 半精度

        # Set Dataloader
        # 通过不同的输入源来设置不同的数据加载方式
        vid_path, vid_writer = None, None
        dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        # 获取类别名字字符串列表
        names = model.module.names if hasattr(model, 'module') else model.names
        # 获取颜色用于预测框
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        # path 图片路径
        # img 经过处理的图片(c,h,w)
        # im0s 原图片尺寸
        # vid _cap
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            # pred[...,0:3] 预测框x y w h中心点+宽高
            # pred[...,4] 置信度得分
            # pred[...,5:-1]为分类概率结果
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            # 后处理 非极大值抑制
            # pred[...,0:3] 预测框xy xy 左上角右下角
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            # 对每一张图片做处理
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # 保存图片路径
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    list1=[]
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            list1.append(f'{names[int(cls)]}')
                            list1.append(f'{conf:.2f}')
                            list1.append(classifyTrash.classify_trash(f'{names[int(cls)]}'))
                            self.output.append(list1)
                            list1=[]
                            im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        print(self.output)
        # output输出类别和置信度
        return self.output


if __name__ == '__main__':
    mydet = Detect(source='testfiles/30.jpg',weights='weights/best_300.pt',conf_thres=0.4)
    list=mydet.detect()

