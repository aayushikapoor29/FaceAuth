import cv2 as cv
from mtcnn import MTCNN 
import os

# for identifying faces
mt = MTCNN()

# automic list of people who have registered
DIR = r'/home/aayushi/fv2/dataset'
registered = os.listdir(DIR)


# register the name with photo
def register_user(username):
    # make the directory of name
    user_folder = os.path.join("dataset", username)
    os.makedirs(user_folder) 

    # intialize camera
    cap = cv.VideoCapture(0)
    count = 0
    print(f"Capturing face for {username}. Please look at the camera...")

    while count < 30:  
        ret, frame = cap.read()

        # nothing is recognized in the camera
        if not ret:
            print("Camera error! Try again.")
            break

        # detect faces
        faces = mt.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']

            face_crop = frame[y:y+h, x:x+w]
            face_resize = cv.resize(face_crop, (200, 200))

            # save the image systematically
            img_path = os.path.join(user_folder, f"{count}.jpg")
            cv.imwrite(img_path, face_resize)
            count += 1
            print(f"Image {count}/30 saved.")

        cv.imshow("Registering...", frame)

        if cv.waitKey(1) == 27: 
            break

    cap.release()
    cv.destroyAllWindows()

    if count == 30:
        print(f"Registration successful for {username}!")
    else:
        # just for unexpected error
        print(f"Registration incomplete. Only {count} images captured.")


if __name__ == "__main__":
    user = input("Enter your Name : ").strip().lower()
    if user not in registered:
        register_user(user)
    else:
        print("User data already exists!!!")
