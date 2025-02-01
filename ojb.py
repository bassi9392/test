import cv2
import time
import random

class GNSS:
    def get_location(self):
        latitude = 37.7749 + random.uniform(-0.1, 0.1)
        longitude = -122.4194 + random.uniform(-0.1, 0.1)
        return latitude, longitude

class StereoCamera:
    def __init__(self, baseline_distance=60.0, focal_length=2.6, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.baseline_distance = baseline_distance
        self.focal_length = focal_length

    def capture_images(self):
        # Capture left and right images
        ret, left_frame = self.cap.read()
        right_frame = None  # Implement capturing right image (if necessary)

        if not ret:
            print("Error: Unable to capture a frame.")
            return None, None

        return left_frame, right_frame

    def calculate_distance(self, bounding_box_size):
        # Estimate distance using the stereo vision formula: distance = baseline * focal_length / disparity
        estimated_distance = (self.baseline_distance * self.focal_length) / bounding_box_size
        return estimated_distance

    def __del__(self):
        self.cap.release()

class ObjectDetection:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            return faces
        else:
            return []

class EngineController:
    def __init__(self):
        self.engine_started = False

    def start_engine(self):
        print("Engine started.")
        self.engine_started = True

    def cut_off_engine(self):
        print("Engine cut-off.")
        self.engine_started = False

def main():
    gnss = GNSS()
    stereo_camera = StereoCamera(baseline_distance=60.0, focal_length=2.6, camera_index=0)  # Stereo Camera using laptop webcam
    object_detection = ObjectDetection()
    engine_controller = EngineController()

    try:
        while True:
            latitude, longitude = gnss.get_location()

            # Capture images from the stereo camera
            left_frame, right_frame = stereo_camera.capture_images()

            # Detect objects (faces)
            detected_faces = object_detection.detect_objects(left_frame)

            # Draw a rectangle and display the object name and distance
            for (x, y, w, h) in detected_faces:
                cv2.rectangle(left_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the object name inside the green rectangle
                object_name = 'Person'  # Replace with the actual object name
                cv2.putText(left_frame, object_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Estimate and display the distance
                bounding_box_size = max(w, h)
                estimated_distance = stereo_camera.calculate_distance(bounding_box_size)
                cv2.putText(left_frame, f"Distance: {estimated_distance:.2f} cm", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Video", left_frame)
            key = cv2.waitKey(1)

            if key == 27:  # Break the loop if 'Esc' key is pressed
                break

            print(f"Location: {latitude}, {longitude}")

            if len(detected_faces) > 0:  # Check if faces are detected
                print(f"{len(detected_faces)} Person(s) Detected")
                if engine_controller.engine_started and estimated_distance < 5:  # Cut-off engine only if the person is below 5 meters
                    print("Object detected within 5 meters. Cutting off engine.")
                    engine_controller.cut_off_engine()
            else:
                print("No Person Detected")

            if not engine_controller.engine_started:
                engine_controller.start_engine()

    except KeyboardInterrupt:
        print("Exiting the program.")

if __name__ == "__main__":
    main()