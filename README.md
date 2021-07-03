**Hướng dẫn chạy**
```python
python yolo_image.py --image images/traffic.jpg --yolo yolo-coco
```
hoặc là cho file video
```python
python yolo_video.py --input videos/traffic_monitor.mp4 --output output/traffic_monitor_out.avi --yolo yolo-coco
```
hoặc là từ webcam
```python
python yolo_video.py --output output/webcam.avi --yolo yolo-coco
```
Trong bài này mình không đi vào lý thuyết mà sử dụng luôn pre-trained model YOLO v3 có sẵn để thực hiện phát hiện vật thể trong ảnh và video.

Đối với bài này các bạn cần tải 3 file sau:
- `coco.names`: chứa tên các class mà YOLO được huấn luyện [link](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- `yolov3.cfg`: configuration file chứa các cài đặt cho YOLO [link](https://pjreddie.com/darknet/yolo/)
- `yolov3.weights`: các pre-trained weights [link](https://pjreddie.com/darknet/yolo/). Do file lớn quá các bạn không upload lên Github được, các bạn tải về theo link mình đính kèm.

Các bước chính khi triển khai pre-trained YOLO v3:
* Load model (cần file weights và configuration)
* Load ảnh, tiền xử lý trước khi đưa vào model
* Lấy tên các output layers, dựa vào đây chúng ta sẽ xác định được các bounding boxes
* Thực hiện Non-max suppression để loại bỏ các bounding boxes chồng chập
* Vẽ các bounding boxes và confidence lên ảnh.

Các bạn có thể xem code chi tiết ở github của mình [YOLO3 Object Detection](https://github.com/huytranvan2010/YOLO3-Object-Detection)

Nhược điểm của YOLOv3:
- Khó phát hiện được các vật thể nhỏ (Faster RCNN làm rất tốt nhưng lại chậm)
- Khó xử lý các vật thể sát nhau

# Tài liệu tham khảo
1. https://www.youtube.com/watch?v=1LCb1PVqzeY
2. https://gilberttanner.com/blog/yolo-object-detection-with-opencv
3. https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
4. https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
5. https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
