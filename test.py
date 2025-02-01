import cv2

# Use 0 instead of 1 for the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load the OpenPose model
net = cv2.dnn.readNetFromTensorflow("pose/coco/pose_iter_440000.caffemodel", "pose/coco/pose_deploy_linevec.prototxt")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform pose estimation
    net.setInput(cv2.dnn.blobFromImage(img, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False))
    out = net.forward()

    # Draw detected keypoints
    for i in range(out.shape[1]):
        confidence = out[0, i, :, 2]
        min_confidence = 0.1  # Minimum confidence threshold
        if confidence > min_confidence:
            # Extract x, y coordinates of keypoints
            x = int(out[0, i, 0, 0] * img.shape[1])
            y = int(out[0, i, 0, 1] * img.shape[0])

            # Draw a circle at the keypoint position
            cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

    # Show the output
    cv2.imshow("Output", img)

    # Check for 'esc' key press
    key = cv2.waitKey(1)
    if key == 27:  # 27 corresponds to the 'esc' key
        break

cap.release()
cv2.destroyAllWindows()
