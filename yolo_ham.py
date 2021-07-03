import numpy as np
import argparse
import cv2
import os
import time 

def extract_boxes_confidences_classIDs(layerOutputs, confidence, W, H):
    # khởi tạo list các detected bounding boxes, confidences và class IDs
    boxes = []
    confidences = []
    classIDs = []   # detected object's class label

    # Duyệt qua các layer outputs
    for output in layerOutputs:
        # duyệt qua each of the detections
        for detection in output:
            # trích xuất the class ID và confidence của the current object detection
            scores = detection[5:]      # bỏ qua 5 cái đầu là x, y, w, h và objectness score, những cái sau là class scores
            classID = np.argmax(scores)
            confidence = scores[classID]

            # chỉ quan tâm predictions nào có confidence đủ lớn (> threshold)
            if confidence > args["confidence"]:
                # scale the bounding box back to the size of the image
                # YOLO trả về  relative size
                box = detection[0:4] * np.array([W, H, W, H])   # element-wise (center_x, center_y, w, h)
                centerX, centerY, width, height = box.astype("int")     # int về cần vẽ pixel

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                # tính toạ độ góc trên bên trái
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bouding boxes, cập nhật list
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    # Vẽ các box vad class text lên ảnh
    # cần đảm bảo có ít nhất 1 detection
    if len(idxs) > 0:
        # duyệt qua các indexes 
        for i in idxs.flatten():
            # lấy ra các thông tin của bounding boxes
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image 

def make_prediction(net, layer_names, LABELS, image, confidence, threshold):
    (H, W) = image.shape[:2]    # lấy height và width
    # construct a blob from the input image and then perform a forward pass of the YOLO
    # object detector, giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1/255., (416, 416), swapRB=True, crop=False)    # tiền xử lý ảnh để tí cho vào YOLO
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layer_names)
    end = time.time()
    # # show timing information on YOLO
    # print("[INFO] YOLO look {:.6f} seconds".format(end - start))

    boxes, confidences, classIDs = extract_boxes_confidences_classIDs(layerOutputs, confidence, W, H)

    # Áp  dụng non-max suppression để loại bỏ bớt các overlapping bounding boxes mà cùng vật thể
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold) 

    return boxes, confidences, classIDs, idxs

if __name__ == '__main__': 
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", required=True, help="base path to the YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    # tải COCO class labels mà YOLO đã trained trên đó
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])     # đặt trong list
    # đọc hết file luôn bằng read() trả về string, dùng strip() để loại bỏ space, tab "\t", xuống dòng "\n" ở đầu
    # và cuối chuỗi (ở giữa không ảnh hưởng, có thể  thêm kí tự vào strip()), sau đó chuyển thành list 
    LABELS = open(labelsPath).read().strip().split("\n")    

    # khởi tạo list các màu để biểu diễn labels
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3))   # mỗi hàng là 1 màu (3 giá trị)

    # lấy đường dẫn đến YOLO weights và model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load YOLO detector đã pre-trained trên COCO dataset (80 classes) bằng OpenCV
    print(["[INFO] loading YOLO from the disk..."])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Lấy layer names của các output layers (chỉ các output layers thôi)
    layer_names = net.getLayerNames()   # tên của tất cả các layers trong YOLO
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]    # getUnconnectedOutLayers() - it gives you the final layers number in the list from, lấy index trừ 1

    # load ảnh rồi đưa vào network
    image = cv2.imread(args["image"])

    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, LABELS, image, args["confidence"], args["threshold"])

    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, COLORS)

    cv2.imshow("image", image)
    cv2.waitKey(0)


"""
https://gilberttanner.com/blog/yolo-object-detection-with-opencv
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/

"""

"""
    YOLO3 sử dụng 3 anchor boxes trong mỗi grid cell. Final output có thể là 13x13, 26x26, 52x52. Do đó ví dụ chọn 13x13
    thì output dầu ra là (13, 13, 3*(4+1+80)).
    Mỗi anchor box là 4 values đầu tiên x, y, w, h tiếp theo là objectness score và cuối cùng là 80 class scores.
"""


