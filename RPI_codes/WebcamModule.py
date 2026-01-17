from picamera2 import Picamera2
import cv2

class Webcam:
    def __init__(self):
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_video_configuration(main={"size": (640, 640)}))
        self.camera.start()

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def get_frame(self):
        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def stop(self):
        self.camera.stop()
        
if __name__ == '__main__':
    webcam = Webcam()
    import time
    try:
        while True:
            time.sleep(1)
            cap.get(cv2.CAP_PROP_FPS)
            frame = webcam.get_frame()
            cv2.imshow("Original Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("exiting program")
    finally:
        webcam.stop()
        cv2.destroyAllWindows()
