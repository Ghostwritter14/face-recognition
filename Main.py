import face_recognition
import cv2
import os
import pickle

# Function to load and encode known faces using deep learning
def load_and_encode_known_faces(data_dir):
    known_face_encodings = []
    known_face_names = []

    for subdir in os.listdir(data_dir):
        name = subdir
        subdir_path = os.path.join(data_dir, subdir)

        for image_name in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image_name)

            # Load image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face using face_recognition
            face_locations = face_recognition.face_locations(rgb_image)

            if len(face_locations) > 0:
                # Use face_recognition library to get encodings
                encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)

                if len(encodings) > 0:
                    face_encoding = encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                else:
                    print(f"No face detected in {image_path}")
            else:
                print(f"No face detected in {image_path}")

    return known_face_encodings, known_face_names

# Function to recognize faces in video
def recognize_faces_in_video(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Match each face with known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face and label the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Load and encode known faces
data_dir = 'Ref_pics' # Or you can name the main directory as you want to
# but please name the subdirectory with the name of the person
known_face_encodings, known_face_names = load_and_encode_known_faces(data_dir)

# Save encodings and names
with open('known_face_encodings.pkl', 'wb') as f:
    pickle.dump(known_face_encodings, f)

with open('known_face_names.pkl', 'wb') as f:
    pickle.dump(known_face_names, f)

# Call the function to start recognizing faces in the video
recognize_faces_in_video(known_face_encodings, known_face_names)

