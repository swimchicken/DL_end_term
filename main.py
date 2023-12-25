import tensorflow as tf
import cv2
import numpy as np

three_feature_model = tf.keras.models.load_model("model/age.h5")
facial_unet_model = tf.keras.models.load_model("model/unet.h5")
facial_vgg_model = tf.keras.models.load_model("model/vgg.h5")

# TODO 測試資料

# image_path = "face2.jpg"
# image = cv2.imread(image_path)
# resized_image = cv2.resize(image, (198, 198))
#
# resized_image = resized_image / 255.0
# resized_image = np.expand_dims(resized_image, axis=0)
# x = four_feature_model.predict(resized_image)
#
#
#
# print(four_feature_model.summary())
# print(facial_unet_model.summary())
# print(facial_vgg_model.summary())

# TODO init face_偵測

do_prediction = True  # 設定是否進行預測

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ID_GENDER_MAP = ['male','female']
ID_RACE_MAP = ['white', 'black', 'asian', 'indian', 'others']

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]

        # 處理年齡問題

        resized_age = cv2.resize(face_roi, (198, 198))
        resized_image = resized_age / 255.0
        resized_age_true = np.expand_dims(resized_image, axis=0)

        # 處理表情問題

        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        resized_face_true = np.expand_dims(resized_face, axis=0)

        # resized_face_roi = cv2.resize(face_roi, (48, 48))
        # resized_face_roi = np.expand_dims(resized_face_roi, axis=0)

        if do_prediction:
            # 表情問題
            predicted_face = facial_vgg_model.predict(resized_face_true)
            predicted_face_squeezed = tf.squeeze(predicted_face, axis=0)
            print("表情: ",predicted_face_squeezed)
            predicted_face_softmax = np.argmax(predicted_face_squeezed)

            # print(predicted_face_softmax)

            emotion_label = label_map[predicted_face_softmax]
            cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)

            # 3種特徵問題
            pred_three = three_feature_model.predict(resized_age_true)
            age_pred = int(pred_three[0] * 100)
            print("年齡: ", age_pred)
            cv2.putText(frame, f"Age: {age_pred} age", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)

            etc = np.argmax(pred_three[1])
            print("膚色: ", etc)
            etc_pre = ID_RACE_MAP[etc]

            cv2.putText(frame, f"Race: {etc_pre}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)

            sex = np.argmax(pred_three[2])
            print("性別: ", sex)
            sex_pre = ID_GENDER_MAP[sex]

            cv2.putText(frame, f"Gender: {sex_pre}", (x, y + h + 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)

    cv2.imshow('Face Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
