from ultralytics import YOLO
import cv2

def count_people(image):
    model = YOLO('yolov5s.pt')
    results = model(image)
    count = sum(1 for result in results if result['class'] == 0)
    return count

photo1 = "example.jpg"
photo2 = "example2.jpg"

count1 = count_people(photo1)
count2 = count_people(photo2)

print(f"사진 1에서 사람 수: {count1}")
print(f"사진 2에서 사람 수: {count2}")
