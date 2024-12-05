from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') 
model.predict(source="0", show=True, stream=True, classes=0) 

image_path_1 = 'example.jpg'
image_path_2 = 'example2.jpg'

results = model([image_path_1, image_path_2])

count =[]

for i in range(len(results)):
    boxes = results[i].boxes 
    masks = results[i].masks 
    keypoints = results[i].keypoints 
    probs = results[i].probs 
    cnt = 0
    
    for box in boxes:
        cnt += 1
    
    cv2.imwrite(f"ex{i}.jpg", results[i].plot())
    cv2.destroyAllWindows()

    count.append(cnt)

print(f"사진 1에서 사람 수: {count[0]}")
print(f"사진 2에서 사람 수: {count[1]}")
print(f"증감 수: {count[0] - count[1]}") if count[0] > count[1] else print(f"증감 수: {count[1] - count[0]}") 
