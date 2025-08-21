import cv2                          # For webcam and image display
import face_recognition             # For face detection and recognition
import os                           # To load images from folder

# -------------------------------
# STEP 1: Load criminal images
# -------------------------------
known_face_encodings = []          # Stores encodings (face data)
known_face_names = []              # Stores criminal names

criminal_folder = "criminals"      # Folder with criminal images

for file_name in os.listdir(criminal_folder):
    image_path = os.path.join(criminal_folder, file_name)
    image = face_recognition.load_image_file(image_path)

    # Convert image to encoding (numerical face data)
    encodings = face_recognition.face_encodings(image)
    if encodings:  # Check if face is found in image
        known_face_encodings.append(encodings[0])
        # Save the name (filename without .jpg)
        known_face_names.append(os.path.splitext(file_name)[0])

# -------------------------------
# STEP 2: Start webcam
# -------------------------------
video = cv2.VideoCapture(0)  # 0 = default webcam

print("Starting Criminal Face Detection... Press 'q' to quit.")

# -------------------------------
# STEP 3: Detect faces in webcam
# -------------------------------
while True:
    ret, frame = video.read()  # Read a frame from webcam
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Find all face locations and encodings in this frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare this face to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display name below face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Criminal Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# STEP 4: Release webcam
# -------------------------------
video.release()
cv2.destroyAllWindows()
