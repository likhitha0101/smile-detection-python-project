import cv2

# ---------- OPEN CAMERA (ROBUST METHOD) ----------
def open_camera():
    for i in range(3):  # try 0,1,2
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            print(f"✅ Camera opened with index {i}")
            return cap
    return None

cap = open_camera()

if cap is None:
    print("❌ ERROR: Webcam not detected")
    exit()

# ---------- LOAD CASCADES ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face_gray = gray[y:y+h, x:x+w]

        # Detect smile
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25)
        )

        # Label
        if len(smiles) > 0:
            label = "Smile 😊"
            color = (0, 255, 0)
        else:
            label = "No Smile 😐"
            color = (0, 0, 255)

        # Draw
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Smile Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()