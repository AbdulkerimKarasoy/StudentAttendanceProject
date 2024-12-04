from flask import Flask, render_template, request, jsonify  # Web framework and utilities
import os  # For interacting with the operating system
from deepface import DeepFace  # DeepFace library for facial recognition
import numpy as np  # For numerical operations
import datetime  # For date and time handling
import csv  # For CSV file handling

# Initialize the Flask application
app = Flask(__name__)

# Folder to store known faces
KNOWN_FACES_DIR = 'known_faces'  # Directory to store registered faces

# File to store attendance records
ATTENDANCE_FILE = 'attendance_records.csv'

# Create the folder if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Dictionary to store known faces embeddings
known_faces = {}

# Function to load known faces from the 'known_faces' directory
def load_known_faces():
    global known_faces
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                # Generate embedding for the face image using DeepFace with Facenet model
                embedding = DeepFace.represent(img_path=img_path, model_name='Facenet')[0]["embedding"]
                # Extract the name from the filename (without extension)
                name = filename.rsplit('.', 1)[0]
                # Store the embedding with the name as key
                known_faces[name] = embedding
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Function to initialize attendance file
def initialize_attendance_file():
    with open(ATTENDANCE_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Date', 'Time'])

# Function to read attendance records
def read_attendance_records():
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                records.append(row)
    return records

# Load known faces and initialize attendance file at the start of the application
load_known_faces()
if not os.path.exists(ATTENDANCE_FILE):
    initialize_attendance_file()

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    attendance_records = read_attendance_records()
    return render_template('index.html', attendance_records=attendance_records)  # Render the main page

# Route to handle registration with captured image from webcam
@app.route('/register_capture', methods=['POST'])
def register_capture():
    # Get the name from the form data
    name = request.form.get('name')
    if not name:
        result = "Please provide a name."
        return jsonify({'result': result})

    # Check if an image file is included
    if 'image_file' not in request.files:
        result = "No image data provided."
        return jsonify({'result': result})

    file = request.files['image_file']
    # Secure the filename and save the file to KNOWN_FACES_DIR
    filename = f"{name}.jpg"
    file_path = os.path.join(KNOWN_FACES_DIR, filename)
    file.save(file_path)

    # Process the captured image using DeepFace
    try:
        # Generate embedding for the face image
        embedding = DeepFace.represent(img_path=file_path, model_name='Facenet')[0]["embedding"]
        known_faces[name] = embedding  # Add to known faces dictionary
        result = f"Face of {name} registered successfully."
    except ValueError:
        # Remove the file if no face is found
        os.remove(file_path)
        result = "No faces found in the image."
    except Exception as e:
        # Remove the file and report the error
        os.remove(file_path)
        result = f"An error occurred: {e}"

    return jsonify({'result': result})

# Route to handle face recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # Temporary directory for recognition files
    TEMP_DIR = 'temp'
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Check if an image file is included
    if 'image_file' not in request.files:
        result = "No image data provided."
        return jsonify({'result': result})

    file = request.files['image_file']
    temp_image_path = os.path.join(TEMP_DIR, 'captured_image.png')
    file.save(temp_image_path)

    # Process the captured image using DeepFace
    try:
        # Generate embedding for the uploaded face image
        uploaded_embedding = DeepFace.represent(img_path=temp_image_path, model_name='Facenet')[0]["embedding"]
    except ValueError:
        result = "No faces found in the image."
        return jsonify({'result': result})
    except Exception as e:
        result = f"An error occurred: {e}"
        return jsonify({'result': result})
    finally:
        # Remove the temporary file after processing
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    # Compare the uploaded embedding with known faces
    name = "Unknown"
    highest_similarity = -1
    threshold = 0.7  # Similarity threshold for recognition
    for person_name, known_embedding in known_faces.items():
        known_embedding = np.array(known_embedding)
        # Compute cosine similarity between embeddings
        similarity = np.dot(uploaded_embedding, known_embedding) / (
            np.linalg.norm(uploaded_embedding) * np.linalg.norm(known_embedding))
        if similarity > highest_similarity:
            highest_similarity = similarity
            name = person_name

    # If the highest similarity is below the threshold, consider as unknown
    if highest_similarity < threshold:
        name = "Unknown"
    if name == 'Unknown':
        result = "Recognition failed"
    else:
        result = f"Hi, {name}! Recognition successful."

    # If recognized successfully, record attendance
    if name != "Unknown":
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        # Write to attendance file
        with open(ATTENDANCE_FILE, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([name, date_str, time_str])

    return jsonify({'result': result})

# Endpoint to get attendance records as JSON
@app.route('/attendance_records', methods=['GET'])
def get_attendance_records():
    attendance_records = read_attendance_records()
    return jsonify({'attendance_records': attendance_records})

# Endpoint to reset attendance records
@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    initialize_attendance_file()
    return jsonify({'result': 'Attendance records have been reset.'})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
