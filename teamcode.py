from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
model.predict(source="0", show=True, stream=True, classes=0)  # [0, 3, 5] for multiple classes


# Define image file names
image_path_1 = 'example.jpg'
image_path_2 = 'example2.jpg'

# Run batched inference on a list of images
results = model([image_path_1, image_path_2], conf = 0.5)  # return a list of Results objects

count =[]

# Process results list
for i in range(len(results)):
    boxes = results[i].boxes  # Boxes object for bbox outputs
    masks = results[i].masks  # Masks object for segmentation masks outputs
    keypoints = results[i].keypoints  # Keypoints object for pose outputs
    probs = results[i].probs  # Class probabilities for classification outputs
    cnt = 0
    
    for box in boxes:
        cnt += 1
    
    # Display result with boxes
    cv2.imwrite(f"ex{i}.jpg", results[i].plot())
    cv2.destroyAllWindows()

    count.append(cnt)

print(f"사진 1에서 사람 수: {count[0]}")
print(f"사진 2에서 사람 수: {count[1]}")
print(f"증감 수: {count[0] - count[1]}") if count[0] > count[1] else print(f"증감 수: {count[1] - count[0]}") 
