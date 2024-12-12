import os
import face_recognition
import pickle

# Path to the dataset
dataset_path = "C:/Users/Lenovo/dataset"
encodings_path = "encodings.pickle"

# Prepare lists to store encodings and labels
known_encodings = []
known_names = []

# Loop through each person in the dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Loop through each image of the person
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        # Get face encodings
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_encodings.append(face_encodings[0])
            known_names.append(person_name)

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print("Encodings saved successfully!")