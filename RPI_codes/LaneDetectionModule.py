import cv2                   
import numpy as np            
import utlis                
from MotorModule import Move  


curveList = []                # Rolling list of recent curve values for smoothing
avgVal = 10                   # Number of frames used to compute moving average for the curve


def getLaneCurve(img, display = 2):
    # Main lane detection pipeline: threshold, warp, histogram, curve calculation and visualization
    imgCopy = img.copy()
    imgResult = img.copy()

    # Step 1: threshold image to isolate lane markings
    imgThres = utlis.thresholding(img)

    # Step 2: warp the thresholded image into bird's eye view using trackbar points
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)  # Show reference points used for perspective transform

    # Step 3: use histogram to find lane center and compute raw curve
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display = True, minPer = 0.5, region = 2)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display = True, minPer = 0.9, region = 1)
    curveRaw = curveAveragePoint - middlePoint

    # Step 4: smooth curve using moving average over last N frames
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))



    # Step 5: build visualization of detected lane and curve overlay if display is enabled
    if display != 0:
       imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3, 0:wT] = 0, 0, 0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
       midY = 450

       # Normalize curve for drawing reference (simple clamping)
       curve2 = curve / 100
       if curve2 > 1: curve2 == 1
       if curve2 < -1: curve2 == -1

       # Draw curve debug info and lane direction lines
       cv2.putText(imgResult, str(curve), (wT//2-80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
       cv2.line(imgResult, (wT//2, midY), (wT//2 + (curve * 3), midY), (255, 0, 255), 5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(
               imgResult,
               (w * x + int(curve // 50), midY-10),
               (w * x + int(curve // 50), midY+10),
               (0, 0, 255),
               2
           )


    if display == 2:
       # Show stacked debug windows: original, warp, histogram and lane overlay
       imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                            [imgHist, imgLaneColor, imgResult]))
       cv2.imshow('ImageStack', imgStacked)
    # elif display == 1:
    #    cv2.imshow('Resutlt', imgResult)



    # Return final curve (steering indicator) and the visualization image
    return curve, imgResult


def process_frame(img):
    # Wrapper used by main driving code: resize frame, update trackbars and get curve
    if img is None:
        print("Empty frame received, skipping processing")
        return None, None

    img = cv2.resize(img, (640, 640))  # Normalize all frames to 640x640 for processing consistency

    initialTrackBarVals = [36, 244, 0, 477]  # Default perspective points for lane region
    utlis.initializeTrackbars(initialTrackBarVals)
    curve, imgResult = getLaneCurve(img, display = 1)
    return curve, imgResult


if __name__ == '__main__':
    # Standalone test loop for lane detection with live camera and motor object
    from time import sleep
    from WebcamModule import Webcam
    webcam = Webcam()
    from MotorModule import Move
    from picarx import Picarx
    px = Picarx()
    motor_controller = Move(px, speed = 0, curve = 0, cam_angle = 0)

    # Initialize trackbars once for the interactive perspective tuning
    initialTrackBarVals = [36, 244, 0, 477]
    utlis.initializeTrackbars(initialTrackBarVals)

    while True:
        # Grab frame from camera and resize to processing resolution
        img = webcam.get_frame()
        img = cv2.resize(img, (640, 640))

        # Run lane detection and display extended debug view
        curve, _ = getLaneCurve(img, display = 2)
        points = utlis.valTrackbars(wT = 640, hT = 640)

        # Quick visualization of the warp configuration
        imgDraw = utlis.drawPoints(img.copy(), points)
        warped = utlis.warpImg(img, points, 640, 640)

        print(curve)

        # Keep car stopped in this demo; only observing the curve value
        motor_controller.move(speed = 0, curve = 0, cam_angle = 0)
        sleep(0.7)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
