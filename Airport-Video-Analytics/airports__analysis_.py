import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "airport.mp4"
cap = cv2.VideoCapture(video_path)

# ---------------------------------------------------------
# Dictionary to store previous positions of detected people
# Used for simple tracking across frames
prev_centers = {}

# Unique ID counter for each detected person
next_id = 0

# Heatmap stores accumulated movement locations
heatmap = None

# Stores how long an object stays nearly stationary (for abandoned bag detection)
static_frames = {}

# Thresholds (can be tuned)
OVER_CROWD_LIMIT = 10        # number of people to call overcrowding
STATIC_PIXEL_THRESHOLD = 2  # very small motion
STATIC_FRAME_LIMIT = 60     # frames (~2 seconds) before bag alert

while True:

    # ---------------------------------------------------------
    # Read next video frame
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame,(640,480))
    h,w = frame.shape[:2]

    # Initialize heatmap on first frame
    if heatmap is None:
        heatmap = np.zeros((h,w), dtype=np.float32)

    # ---------------------------------------------------------
    # YOLO detects objects in current frame
    # We only care about "person" and "backpack/handbag/suitcase"
    results = model(frame, stream=True)

    people_count = 0
    current_centers = {}

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            # Interested classes for video analytics
            if label not in ["person", "backpack", "handbag", "suitcase"]:
                continue

            # Bounding box coordinates
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx = (x1+x2)//2
            cy = (y1+y2)//2

            # ---------------------------------------------------------
            # Count people
            if label == "person":
                people_count += 1

            # ---------------------------------------------------------
            # Simple centroid-based tracking
            assigned = False
            for pid,(px,py) in prev_centers.items():
                if abs(cx-px)+abs(cy-py) < 40:
                    current_centers[pid]=(cx,cy)
                    obj_id = pid
                    assigned=True
                    break

            if not assigned:
                current_centers[next_id]=(cx,cy)
                obj_id = next_id
                next_id+=1

            # ---------------------------------------------------------
            # Heatmap accumulation (add full bounding box area)
            heatmap[y1:y2, x1:x2] += 1

            # ---------------------------------------------------------
            # Abandoned bag detection (stationary object logic)
            if obj_id not in static_frames:
                static_frames[obj_id] = 0

            dx = 0
            dy = 0
            if obj_id in prev_centers:
                dx = cx-prev_centers[obj_id][0]
                dy = cy-prev_centers[obj_id][1]

            if abs(dx)<STATIC_PIXEL_THRESHOLD and abs(dy)<STATIC_PIXEL_THRESHOLD:
                static_frames[obj_id]+=1
            else:
                static_frames[obj_id]=0

            abandoned = static_frames[obj_id]>STATIC_FRAME_LIMIT and label!="person"

            # ---------------------------------------------------------
            # Draw bounding boxes
            color=(0,255,0)
            if abandoned:
                color=(0,0,255)   # red box for abandoned object

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            if abandoned:
                cv2.putText(frame,"ABANDONED",(x1,y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    # Update tracked centers
    prev_centers = current_centers.copy()

    # ---------------------------------------------------------
    # Overcrowding detection
    crowd_status = "NORMAL"
    if people_count > OVER_CROWD_LIMIT:
        crowd_status = "OVER CROWDED"

    cv2.putText(frame,f"People: {people_count}  Status: {crowd_status}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,0),2)

    # ---------------------------------------------------------
    # Heatmap visualization
    heatmap_blur = cv2.GaussianBlur(heatmap,(25,25),0)
    heat_norm = cv2.normalize(heatmap_blur,None,0,255,cv2.NORM_MINMAX)
    heat_color = cv2.applyColorMap(heat_norm.astype(np.uint8),cv2.COLORMAP_JET)

    cv2.imshow("YOLO Video Analytics", frame)
    cv2.imshow("People Density Heatmap", heat_color)

    # Press q to quit
    if cv2.waitKey(30)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)

track_window = None
roi_hist = None

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

heatmap = None
frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))
    h,w = frame.shape[:2]

    if heatmap is None:
        heatmap = np.zeros((h,w), dtype=np.float32)

    # -------- Background subtraction --------
    fg = mog.apply(frame)
    fg = cv2.medianBlur(fg,5)
    _, fg = cv2.threshold(fg,200,255,cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    people = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        x,y,w1,h1 = cv2.boundingRect(cnt)
        cx = x+w1//2
        cy = y+h1//2

        people+=1

        heatmap[cy,cx]+=1

        cv2.rectangle(frame,(x,y),(x+w1,y+h1),(0,255,0),2)

    # -------- Density --------
    if people>8:
        density="HIGH"
    elif people>4:
        density="MEDIUM"
    else:
        density="LOW"

    cv2.putText(frame,f"People: {people} Density: {density}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,0),2)

    # -------- CamShift Tracking --------
    if track_window is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        cv2.polylines(frame,[pts],True,(0,255,255),2)

    # -------- Heatmap Visualization --------
    heat_norm = cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX)
    heat_color = cv2.applyColorMap(heat_norm.astype(np.uint8),cv2.COLORMAP_JET)

    cv2.imshow("Video Analytics", frame)
    cv2.imshow("Motion Heatmap", heat_color)

    key=cv2.waitKey(30)&0xFF

    if key==ord('q'):
        break

    # press t to select person
    elif key==ord('t'):
        roi = cv2.selectROI("Select Person", frame, fromCenter=False)
        cv2.destroyWindow("Select Person")

        if roi!=(0,0,0,0):
            x,y,w1,h1=roi
            track_window=(int(x),int(y),int(w1),int(h1))

            hsv_roi=cv2.cvtColor(frame[y:y+h1,x:x+w1],cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
            roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            print("Tracking started")

cap.release()
cv2.destroyAllWindows()

