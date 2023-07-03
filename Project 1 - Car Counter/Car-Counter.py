from ultralytics import YOLO
import cv2
import cvzone
from sort import *


# https://github.com/abewley/sort/blob/master/sort.py

# Đọc video và load model
cap = cv2.VideoCapture("../Videos/traffic.mp4")

model = YOLO("C:\\Users\\Administrator\\PycharmProjects\\Object-Detection-YOLO\\YOLO - Weights\\yolov8n.pt")
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table",
    "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread("r.png")

# tracking
w_norm = 800
h_norm = 550
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
limits_left = [int(0.19*w_norm), int(0.45*h_norm), int(0.49*w_norm), int(0.45*h_norm)]
limits_right = [int(0.5*w_norm), int(0.6*h_norm), int(w_norm), int(0.6*h_norm)]
totalCount_left = []
totalCount_right = []


# Đọc từng frame của video một cách liên tục
while True:
    _, frame = cap.read()
    resize_mask = cv2.resize(mask, (w_norm, h_norm))
    resize_fr = cv2.resize(frame, (w_norm, h_norm))
    imgRegion = cv2.bitwise_and(resize_fr, resize_mask)

    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

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
        boxes = r.boxes.data

        # lọc ra box chứa car, bus, truck
        motor = boxes[(boxes[:, -1] == 2.0) | (boxes[:, -1] == 5.0) | (boxes[:, -1] == 7.0)].numpy()
        detections = np.vstack((detections, motor[:, :5]))  # x1, y1, x2, y2, conf

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    resultTracker = tracker.update(detections)  # x1, y1, x2, y2, id

    cv2.line(resize_fr, pt1=(limits_left[0], limits_left[1]), pt2=(limits_left[2], limits_left[3]), color=(0, 0, 255), thickness=3)
    cv2.line(resize_fr, pt1=(limits_right[0], limits_right[1]), pt2=(limits_right[2], limits_right[3]), color=(0, 0, 255), thickness=3)

    # Vẽ bounding box và đến số vật thể ko trùng nhau
    for result in resultTracker:
        x1, y1, x2, y2, id = [int(x) for x in result]
        w, h = x2 - x1, y2 - y1

        # Vẽ bounding box
        cvzone.cornerRect(resize_fr, (x1, y1, w, h), l=9, colorR=(255, 0, 0))
        cvzone.putTextRect(resize_fr, f"{int(id)}", (max(0, x1), max(35, y1)),
                           scale=1.1, thickness=1, offset=3)

        # Tìm và vẽ tọa độ center point của bounding box
        cx = int(x1 + w/2)
        cy = int(y1 + h/2)
        cv2.circle(resize_fr, (cx, cy), 5, (255, 0, 255), thickness=cv2.FILLED)

        # Nếu tâm vật thể chạm tới vùng xác định trên ảnh
        if limits_left[0] < cx < limits_left[2] and limits_left[1] - 15 < cy < limits_left[3] + 15:
            # Nếu vật đó chưa được đếm thì tính thêm 1
            if totalCount_left.count(id) == 0:
                totalCount_left.append(id)
                cv2.line(resize_fr, pt1=(limits_left[0], limits_left[1]), pt2=(limits_left[2], limits_left[3]),
                         color=(0, 255, 0), thickness=3)

        if limits_right[0] < cx < limits_right[2] and limits_right[1] - 15 < cy < limits_right[3] + 15:
            # Nếu vật đó chưa được đếm thì tính thêm 1
            if totalCount_right.count(id) == 0:
                totalCount_right.append(id)
                cv2.line(resize_fr, pt1=(limits_right[0], limits_right[1]), pt2=(limits_right[2], limits_right[3]),
                         color=(0, 255, 0), thickness=3)

    cvzone.putTextRect(resize_fr, f"Count: {len(totalCount_left)}", (0, 50),
                       colorR=(255, 255, 255), colorT=(0, 0, 0), scale=1.1, thickness=1, offset=5)

    cvzone.putTextRect(resize_fr, f"Count: {len(totalCount_right)}", (510, 50),
                       colorR=(255, 255, 255), colorT=(0, 0, 0), scale=1.1, thickness=1, offset=5)

    cv2.imshow("Image", resize_fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
