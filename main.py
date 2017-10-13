#!/usr/bin/python3
#encoding: utf-8
import cv2
from urllib import request
import numpy as np
from PIL import Image

ESC = 27
cascade_path = "/home/c0115114ca/my/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

serval = Image.open("sab.png")

# stream start
stream = request.urlopen("http://10.201.35.61:4747/mjpegfeed")

print("なにこれー！")
print("すごーい！")

bytes = b''
cnt = 0
facerect = []
while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8') # jpg start point
    b = bytes.find(b'\xff\xd9') # jpg end point
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if cnt % 10 == 0: # 10フレームに1回
            # 顔検知
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
        cnt += 1

        servaled_img = None
        if len(facerect) > 0:
            imgpil = Image.fromarray(img[:, :, ::-1].copy())
            for rect in facerect:
                x, y, w, h = rect[:4]
                ww = int(w * 1.3)
                hh = int(h * 3)
                # アニメ顔を人体にフィットさせる
                resized_serval = serval.resize((ww, hh))
                imgpil.paste(resized_serval, (x - abs(w - ww) // 2, y - int(abs(h - hh) / 1.5)), resized_serval)
            servaled_img = np.asarray(imgpil)
            servaled_img = servaled_img[:, :, ::-1].copy()

        # 画像表示
        cv2.imshow("aiueo", img if servaled_img is None else servaled_img)
        if cv2.waitKey(1) == ESC:
            exit(0)
