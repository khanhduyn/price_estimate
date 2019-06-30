import face_recognition
import glob
import os
import numpy as np

known_face_encodings = []
known_face_names = []

def load_image(path):
    for img in glob.glob(path + '/*.jpg'):
        name = os.path.basename(img).split('.')[0]
        print('User loading ...', name)
        known_face_names.append(name)
        image = face_recognition.load_image_file(img)
        image_encode = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(image_encode)

def identify(path):
    input_img = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(input_img)
    face_encodings = face_recognition.face_encodings(input_img, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        print('face distance: ', face_distances)
        # print('face matches: ', matches)
        if face_distances[best_match_index] <= 0.5:
            name = known_face_names[best_match_index]

        face_names.append(name)

    print(face_names)
    return face_names[0]

if __name__ == "__main__":
    print("Test Identify")
    load_image('known_face')
    name = identify('./test_img/test.jpg')
    print(name)

