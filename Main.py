import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
pTime = 0
mpdraw = mp.solutions.drawing_utils
mpfacmash = mp.solutions.face_mesh
facemesh = mpfacmash.FaceMesh(max_num_faces=2)
draw_specs = mpdraw.DrawingSpec(thickness=1, circle_radius=2)
while True:
    succes, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(image)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img, faceLms, mpfacmash.FACE_CONNECTIONS,
                                  draw_specs,draw_specs)
            # for lms in faceLms.landmarks:
                # print(lm)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow('HACKER HANDS', img)
    cv2.waitKey(5)