import cv2 as cv
import os
from mtcnn import MTCNN
import numpy as np
from commons import validate_username, registered, mt, DIR


def realtime_face_details():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        faces = mt.detect_faces(frame)
        x, y, w, h = faces[0]['box']
        face = frame[y:y+h, x:x+w]
        current_face = cv.resize(face, (200, 200))
        break   
    cap.release()
    cv.destroyAllWindows()
    return current_face


def compute_average(user_folder):
    images = []
    for img_file in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_file)
        img = cv.imread(img_path)
        img = cv.resize(img, (200,200))
        images.append(img)
    avg_image = np.mean(images, axis=0).astype(np.uint8)
    return avg_image


def authenticate(user):
    # getting details of all the photos
    user_folder = os.path.join(DIR, user)
    print(f"Authenticating {user}")

    # get current face
    current_face = realtime_face_details()

    # compute the average face
    avg_image = compute_average(user_folder)

    # compute error and check
    error = np.mean((current_face.astype("float") - avg_image.astype("float")) ** 2)

    threshold = 1000
    if error<threshold:
        print("authentication successful")
    else:
        print("face does not match")

    



if __name__ == "__main__":
    name = input("Enter your name: ").lower()
    if validate_username(name):
        if name in registered:
            authenticate(name)
        else:
            print("User doesn't exist, register first.")