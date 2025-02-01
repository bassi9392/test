import cv2

# Placeholder values, replace with actual values
camera_fov_horizontal_deg = 60.0  # Camera's horizontal field of view in degrees
object_real_height_meters = 1.7  # Real-world height of the detected object in meters

thres = 0.45  # Threshold to detect object

# Use the specified video file instead of the default camera
video_path = r'video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file is successfully opened
if not cap.isOpened():
    print(f"Error: Unable to open the video file at {video_path}")
    exit()
32  
classNames = []
classFile = 'coco.names'  # Use straight quotes here
try:
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: {classFile} not found. Make sure the file exists.")
    exit()

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def is_person_close(distance):
    return distance < 5.0  # Cut-off the engine when the person is below 5 meters

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from video")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Assuming you have information about the real-world size of the object
            object_height_pixels = box[3]
            object_distance_meters = (object_real_height_meters * cap.get(4)) / (2 * object_height_pixels * (1 / 2 * 3.14 * (camera_fov_horizontal_deg / 360)))

            if is_person_close(object_distance_meters):
                print("Person is below 5 meters.")
                print("initiating Engine cut-off.")
                print("cutting-off Engine.")
                # Add code to control the engine (cut-off) here

    cv2.imshow("Output", img)

    # Check for 'esc' key press
    key = cv2.waitKey(1)
    if key == 27:  # 27 corresponds to the 'esc' key
        break

cap.release()
cv2.destroyAllWindows()