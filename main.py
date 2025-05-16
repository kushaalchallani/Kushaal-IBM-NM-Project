import whisper
import sounddevice as sd
import numpy as np
import warnings
import torch
import cv2
import face_recognition
import sqlite3
from datetime import datetime

warnings.filterwarnings('ignore')

SAMPLE_RATE = 16000
RECORD_DURATION = 5
DATABASE_NAME = 'emergency_medical.db'
TOLERANCE = 0.5
EMERGENCY_PROTOCOLS = {
    "heart attack": {
        "actions": ["Call emergency services", "Have the patient sit down", "Chew an aspirin (if available)"],
        "advice": "Stay calm, don’t move too much",
        "follow_up": "Visit a heart specialist"
    },
    "headache": {
        "actions": ["Sit the patient down", "Give water", "Avoid bright lights and loud sounds"],
        "advice": "Rest in a quiet place",
        "follow_up": "See a doctor if pain lasts"
    },
    "eye pain": {
        "actions": ["Avoid rubbing the eyes", "Wash with clean water", "Apply a cold compress"],
        "advice": "Don’t use eye drops unless prescribed",
        "follow_up": "Visit an eye doctor"
    },
    "stomach ache": {
        "actions": ["Have the patient rest", "Give water", "Avoid heavy food"],
        "advice": "Apply heat to the stomach area",
        "follow_up": "See a doctor if pain persists"
    },
    "fever": {
        "actions": ["Give water", "Rest the patient", "Keep the room cool"],
        "advice": "Use a cold compress to lower temperature",
        "follow_up": "Visit a doctor if fever lasts more than 3 days"
    },
    "dizziness": {
        "actions": ["Have the patient sit down", "Give water", "Keep them cool"],
        "advice": "Don’t let the patient stand up quickly",
        "follow_up": "See a doctor if dizziness continues"
    },
    "back pain": {
        "actions": ["Have the patient lie down", "Apply a cold compress", "Support the back if standing"],
        "advice": "Avoid bending or lifting heavy objects",
        "follow_up": "Consult a doctor if pain doesn’t improve"
    },
    "cough": {
        "actions": ["Keep the patient hydrated", "Rest the patient", "Avoid smoking or strong smells"],
        "advice": "Cover mouth when coughing",
        "follow_up": "Visit a doctor if cough lasts for more than 2 weeks"
    },
    "rash": {
        "actions": ["Keep the skin cool", "Apply soothing lotion", "Avoid scratching the rash"],
        "advice": "Don’t expose the rash to sunlight",
        "follow_up": "See a doctor if rash spreads or worsens"
    },
    "nausea": {
        "actions": ["Have the patient sit down", "Give water", "Avoid strong smells or food"],
        "advice": "Try to rest until the feeling passes",
        "follow_up": "See a doctor if nausea persists"
    }
}

def setup_database():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (face_id TEXT PRIMARY KEY,
                  face_encoding BLOB,
                  symptoms TEXT,
                  last_updated TEXT,
                  encounter_count INTEGER)''')
    conn.commit()
    conn.close()

def record_symptoms():
    print("\n🔴 Describe emergency (5 seconds)...")
    try:
        audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)

        if np.max(np.abs(audio)) < 0.01:
            print("⚠️ No audio detected, restarting recording...")
            return record_symptoms()

        model = whisper.load_model("tiny")
        result = model.transcribe(audio, fp16=False)
        text = result["text"].lower()
        print(f"\n🔊 Detected: '{text}'")

        corrected_text = text
        corrections = {
            "chest been with": "chest pain",
            "i have just pain": "chest pain",
            "i have a chest been with": "chest pain"
        }

        for incorrect, correct in corrections.items():
            corrected_text = corrected_text.replace(incorrect, correct)

        symptoms = [s for s in EMERGENCY_PROTOCOLS if s in corrected_text]
        return symptoms if symptoms else None

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

def get_known_faces():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT face_id, face_encoding FROM patients")
    records = c.fetchall()
    conn.close()

    face_data = []
    for face_id, encoding_blob in records:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        face_data.append((face_id, encoding))
    return face_data

def scan_face():
    print("\n📸 Scanning face...")
    cap = cv2.VideoCapture(0)
    face_id = None
    current_encoding = None
    try:
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if not encodings:
                    continue

                current_encoding = encodings[0]
                known_faces = get_known_faces()

                for stored_id, stored_encoding in known_faces:
                    matches = face_recognition.compare_faces([stored_encoding], current_encoding, tolerance=TOLERANCE)
                    if matches[0]:
                        face_id = stored_id
                        print("\n✅ Existing patient recognized")
                        return face_id, current_encoding

                face_id = str(abs(hash(current_encoding.tobytes())))
                print("\n🆕 New patient detected")
                return face_id, current_encoding

            cv2.imshow('Position face (Press Q to skip)', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        print("⚠️ Face not detected")
        return None
    finally:
        cap.release()
        cv2.destroyAllWindows()

def get_patient_history(face_id, current_symptoms):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT symptoms, encounter_count FROM patients WHERE face_id=?", (face_id,))
    record = c.fetchone()
    conn.close()

    if record:
        previous_symptoms = record[0].split(', ') if record[0] else []
        encounter_count = record[1]

        print("\n=== MEDICAL HISTORY ===")
        print(f"Previous symptoms: {', '.join(previous_symptoms)}")
        print(f"Total previous visits: {encounter_count}")

        recurring = set(current_symptoms) & set(previous_symptoms)
        if recurring:
            print(f"\n⚠️ RECURRING CASE: Patient had similar symptoms before ({', '.join(recurring)})")
        else:
            print("\nNo previous cases of these exact symptoms")
        print("=" * 30)

def update_records(face_id, face_encoding, symptoms):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()

    c.execute("SELECT symptoms FROM patients WHERE face_id=?", (face_id,))
    existing = c.fetchone()

    if existing:
        new_symptoms = f"{existing[0]}, {', '.join(symptoms)}"
        c.execute('''UPDATE patients 
                     SET symptoms=?, last_updated=?, encounter_count=encounter_count+1 
                     WHERE face_id=?''',
                  (new_symptoms, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), face_id))
    else:
        c.execute('''INSERT INTO patients 
                     (face_id, face_encoding, symptoms, last_updated, encounter_count)
                     VALUES (?, ?, ?, ?, 1)''',
                  (face_id, face_encoding.tobytes(),
                   ', '.join(symptoms),
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def provide_care(symptoms):
    print("\n🚨 EMERGENCY PROTOCOLS 🚨")
    for symptom in symptoms:
        protocol = EMERGENCY_PROTOCOLS[symptom]
        print(f"\n❗ {symptom.upper()}:")
        print("➤ ACTIONS:")
        for action in protocol["actions"]:
            print(f"  • {action}")
        print(f"\n➤ ADVICE: {protocol['advice']}")
        print(f"➤ FOLLOW-UP: {protocol['follow_up']}")

def emergency_system():
    setup_database()
    print("\n=== 🚑 EMERGENCY MEDICAL ASSISTANT ===")

    while True:
        symptoms = record_symptoms()
        if not symptoms:
            continue

        face_data = scan_face()
        if face_data:
            face_id, face_encoding = face_data
            get_patient_history(face_id, symptoms)
            update_records(face_id, face_encoding, symptoms)

        provide_care(symptoms)

        if input("\nPress Enter to continue or Q to quit: ").strip().lower() == 'q':
            break

if __name__ == "__main__":
    try:
        import sounddevice as sd
        import whisper
        import numpy as np
        import torch
        import face_recognition
        import sqlite3
        import cv2

        devices = sd.query_devices()
        if not any(dev['max_input_channels'] > 0 for dev in devices):
            print("❌ No microphone detected")
            exit()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            exit()
        cap.release()

        emergency_system()

    except Exception as e:
        print(f"❌ System error: {e}")
