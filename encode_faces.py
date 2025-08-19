# encode_faces.py
import face_recognition
import os
import pickle

print("Starting face encoding process...")

# Dataset folder ka path
KNOWN_FACES_DIR = 'dataset'
# Model jo use karna hai: 'hog' (CPU ke liye tez) ya 'cnn' (GPU ke liye aacha)
MODEL = 'hog' # Keep 'hog' for CPU efficiency, change to 'cnn' if you have a strong GPU and need higher accuracy.

# Encodings aur naam store karne ke liye lists
known_faces_encodings = []
known_faces_names = []

# Dataset folder ke har folder (har व्यक्ति) ke liye loop chalana
for name in os.listdir(KNOWN_FACES_DIR):
    # Us व्यक्ति ke folder ka path
    person_dir_path = os.path.join(KNOWN_FACES_DIR, name)
    
    # Agar yeh ek folder hai
    if os.path.isdir(person_dir_path):
        print(f"Processing images for: {name}")
        
        # Us folder ke andar har image ke liye loop chalana
        for filename in os.listdir(person_dir_path):
            # Image file ka path
            image_path = os.path.join(person_dir_path, filename)
            
            # Check if the file is an image (simple check based on extension)
            if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))):
                print(f"  - Skipping non-image file: {filename}")
                continue

            # Image ko load karna
            try:
                image = face_recognition.load_image_file(image_path)
                
                # Image me chehre ke encodings nikalna
                # Har image me ek hi chehra mankar chal rahe hain
                # Use the specified MODEL (hog or cnn)
                encodings = face_recognition.face_encodings(image, model=MODEL)
                
                if encodings:
                    # Pehla encoding (chehra) lena
                    encoding = encodings[0]
                    
                    # Encoding aur naam ko list me add karna
                    known_faces_encodings.append(encoding)
                    known_faces_names.append(name)
                    print(f"  - Encoded {filename} for {name}")
                else:
                    print(f"  - WARNING: No face found in {filename}. Skipping.")
            
            except Exception as e:
                print(f"  - ERROR: Could not process {filename}. Reason: {e}")

# Encodings aur naamo ko ek file me save karna
# Changed filename to encodings.pickle to match app.py
ENCODINGS_FILE = "encodings.pickle" 
print(f"\nSaving encodings to '{ENCODINGS_FILE}'...")
data = {"encodings": known_faces_encodings, "names": known_faces_names}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("\nEncoding complete and data saved successfully!")
print(f"Total {len(known_faces_encodings)} faces encoded for {len(set(known_faces_names))} people.")
