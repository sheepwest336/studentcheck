from ultralytics import YOLO
import cv2
import numpy as np


def count_people(image_path):
    model = YOLO('yolov8n.pt') 
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    results = model(image)
    
    count = sum(1 for result in results[0].boxes if result.cls == 0)
    return count

photo1 = "example.jpeg"
photo2 = "example.jpeg"

try:
    count1 = count_people(photo1)
    count2 = count_people(photo2)

    print(f"사진 1에서 사람 수: {count1}")
    print(f"사진 2에서 사람 수: {count2}")
    print(f"증감 수: {count2 - count1}")

    image1 = cv2.imread(photo1)
    image2 = cv2.imread(photo2)
    blob1 = cv2.dnn.blobFromImage(image1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=True)
    blob2 = cv2.dnn.blobFromImage(image2, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=True)

    cv2.imshow("Original Photo 1", image1)
    cv2.imshow("Original Photo 2", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"오류 발생: {e}")
