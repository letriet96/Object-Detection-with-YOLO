from ultralytics import YOLO
import cv2
import cvzone
import numpy as np

# https://github.com/abewley/sort/blob/master/sort.py

# Đọc video và load model
cap = cv2.VideoCapture("../Videos/mask.mp4")

model = YOLO("C:\\Users\\Administrator\\PycharmProjects\\Object-Detection-YOLO\\YOLO - Weights\\ppe.pt")
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
           'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# tracking
w_norm = 400
h_norm = 600

# Đọc từng frame của video một cách liên tục
while True:
    _, frame = cap.read()
    resize_fr = cv2.resize(frame, (w_norm, h_norm))

    result = model(resize_fr, stream=True)
    detections = np.empty((0, 6))

    # lấy ra danh sách các box chứa vật thể cần detect và lưu nó vào detections
    for r in result:
        """
        ultralytics.yolo.engine.results.Boxes object with attributes:
        data: tensor([[110.6327, 193.4959, 143.0679, 217.1768,   0.8208,   2.0000]]) 
        -> x1, y1, x2, y2, conf, cls
        cls: tensor([2.])
        conf: tensor([0.8208])
        id: None
        is_track: False
        orig_shape: (400, 600)
        shape: torch.Size([1, 6])
        xywh: tensor([[126.8503, 205.3363,  32.4352,  23.6808]])
        xywhn: tensor([[0.2114, 0.5133, 0.0541, 0.0592]])                
        xyxy: tensor([[110.6327, 193.4959, 143.0679, 217.1768]])
        xyxyn: tensor([[0.1844, 0.4837, 0.2384, 0.5429]])
        """
        boxes = r.boxes.data  # <class 'torch.Tensor'>
        boxes = boxes[((boxes[:, -1] == 1.0) | (boxes[:, -1] == 3.0)) & (boxes[:, -2] > 0.3)]

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cls = int(cls)
            classes_name = classes[cls]
            mycolor = None

            if classes_name in ['Mask']:
                mycolor = (0, 255, 0)

            else:
                mycolor = (0, 0, 255)
            cvzone.cornerRect(resize_fr, (x1, y1, w, h), colorR=mycolor)
            cvzone.putTextRect(resize_fr, f"{classes_name}", (max(0, x1), max(35, y1)), colorT=mycolor, scale=1,
                               thickness=1)

    cv2.imshow("Image", resize_fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
