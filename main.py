import tensorflow as tf
import cv2

four_feature_model = tf.keras.models.load_model("model/age.h5")
facial_unet_model = tf.keras.models.load_model("model/unet.h5")
facial_vgg_model = tf.keras.models.load_model("model/vgg.h5")

print(four_feature_model.summary())
print(facial_unet_model.summary())
print(facial_vgg_model.summary())

# init face_偵測

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]

        predicted_age = four_feature_model.predict(face_roi)
        cv2.putText(frame, f"Age: {predicted_age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow('Face Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
