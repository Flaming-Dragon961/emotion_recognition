from deepface import DeepFace
import cv2
from collections import deque, Counter

avg = deque(maxlen=15)

images = {
    "happy": "img/happy.jpg",
    "sad": "img/sad.png",
    "fear": "img/fear.png",
    "neutral": "img/neutral.png",
    "surprise": "img/surprise.png",
    "angry": "img/angry.jpg"
}

camera = cv2.VideoCapture(0)

count = 0
emotion = "neutral"
while True:
    read,frame = camera.read()
    if not read:
        break
    count+=1
    try:
        if count % 3 == 0:
            results = DeepFace.analyze(frame,actions=["emotion"],enforce_detection=False,detector_backend="opencv")
            if isinstance(results, dict):
                emotion = results["dominant_emotion"]
            else:
                emotion = results[0]["dominant_emotion"]
            avg.append(emotion)
            emotion = Counter(avg).most_common(1)[0][0]
        cv2.putText(frame, emotion, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to quit", (10, 470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        emotion_img_path = images.get(emotion, "img/neutral.png")
        emotion_img = cv2.imread(emotion_img_path)
        frame_resized = cv2.resize(frame, (640, 480))
        if emotion_img is not None:
            emotion_img = cv2.resize(emotion_img, (300, 480))
            side_by_side = cv2.hconcat([frame_resized, emotion_img])
            cv2.imshow("Face Emotion Recognition", side_by_side)
        else:
            cv2.imshow("Face Emotion Recognition", frame_resized)

    except Exception as e:
        cv2.imshow("Face Emotion Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
