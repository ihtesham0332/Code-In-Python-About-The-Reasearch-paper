import cv2
import numpy as np
from openvino.inference_engine import IECore
import face_recognition
import os

# Initialize the Inference Engine
ie = IECore()

# Load the face detection model
model_xml = "path/to/face-detection-adas-0001.xml"
model_bin = "path/to/face-detection-adas-0001.bin"
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="GPU")

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# Read and pre-process an input image
n, c, h, w = net.input_info[input_blob].input_data.shape

# Load known faces
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

# Load known faces
known_face_encodings = []
known_face_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        image = cv2.imread(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []

def main():
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Error: Could not open video stream from camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("Error: Could not read frame from camera.")

            small_frame = cv2.resize(frame, (w, h))
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            input_image = np.expand_dims(rgb_small_frame.transpose(2, 0, 1), 0)

            # Run the inference
            res = exec_net.infer(inputs={input_blob: input_image})

            # Process output
            res = res[out_blob]
            face_locations = []
            for obj in res[0][0]:
                if obj[2] > 0.5:
                    xmin = int(obj[3] * frame.shape[1])
                    ymin = int(obj[4] * frame.shape[0])
                    xmax = int(obj[5] * frame.shape[1])
                    ymax = int(obj[6] * frame.shape[0])
                    face_locations.append((ymin, xmax, ymax, xmin))

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), FRAME_THICKNESS)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), FONT_THICKNESS)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except IOError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
