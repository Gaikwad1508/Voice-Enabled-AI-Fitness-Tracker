import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import cvzone
import speech_recognition as sr
import time

# Initialize the YOLO model and video capture
model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

# Initialize variables
count = 0
up_thresh = 150
down_thresh = 90
pushup_up_left = False
pushup_up_right = False
combine = False
left_hand_counter = 0
right_hand_counter = 0
combine_counter = 0

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('voice', voices[1].id)  # Use female voice
speech_queue = queue.Queue(maxsize=10)  # Prevent speech queue overflow

# Thread-safe mode management
mode_lock = threading.Lock()
mode = None

def set_mode(new_mode):
    global mode
    with mode_lock:
        mode = new_mode

def get_mode():
    with mode_lock:
        return mode

# Speech function
def speak(text):
    if not speech_queue.full():  # Avoid queue overflow
        speech_queue.put(text)

def worker_speak():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit condition for the thread
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Listen for voice commands
def listen_commands():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except OSError as e:
        print(f"Microphone error: {e}")
        return

    while True:
        if get_mode() == 'stop':
            print("Stopping command listener.")
            break
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print('Listening...')
                audio = recognizer.listen(source, timeout=5)  # Timeout prevents indefinite block
                commands = recognizer.recognize_google(audio).lower()
                print(commands)
                if 'normal' in commands:
                    speak('Normal mode started')
                    set_mode('normal')
                elif 'combine' in commands:
                    speak('Combine mode started')
                    set_mode('combine')
                elif 'stop' in commands:
                    speak('Stopping...')
                    set_mode('stop')
        except sr.UnknownValueError:
            print('Could not understand the audio.')
        except sr.RequestError as e:
            print(f"Speech recognition API error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        time.sleep(0.1)  # Prevent excessive CPU usage

# Calculate angle between three points
def calculate_angle(px1, py1, px2, py2, px3, py3):
    a = np.sqrt((px2-px1)**2 + (py2-py1)**2)
    b = np.sqrt((px3-px2)**2 + (py3-py2)**2)
    c = np.sqrt((px3-px1)**2 + (py3-py1)**2)
    angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    return np.degrees(angle)

# Start threads
thread_speech = threading.Thread(target=worker_speak, daemon=True)
thread_speech.start()

thread_listen = threading.Thread(target=listen_commands, daemon=True)
thread_listen.start()

# Main loop for video capture and processing
while True:
    ret, frame = cap.read()
    if not ret or get_mode() == 'stop':  # Exit on stop command
        break
    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 2 != 0:  # Skip alternate frames for efficiency
        continue
    
    # Make predictions
    result = model.track(frame)
    
    if result[0].keypoints is not None:  # Ensure keypoints exist
        keypoints = result[0].keypoints.xy.cpu().numpy()
        for keypoint in keypoints:
            if len(keypoint) < 11:  # Ensure minimum keypoints for arms
                continue

            # Draw keypoints on the frame
            for i, point in enumerate(keypoint):
                cx, cy = int(point[0]), int(point[1])
                cvzone.putTextRect(frame, f'{i}', (cx, cy), 1, 2)
            
            # Extract arm keypoints
            cx1, cy1 = int(keypoint[5][0]), int(keypoint[5][1])
            cx2, cy2 = int(keypoint[7][0]), int(keypoint[7][1])
            cx3, cy3 = int(keypoint[9][0]), int(keypoint[9][1])
            cx4, cy4 = int(keypoint[6][0]), int(keypoint[6][1])
            cx5, cy5 = int(keypoint[8][0]), int(keypoint[8][1])
            cx6, cy6 = int(keypoint[10][0]), int(keypoint[10][1])

            left_hand_angle = calculate_angle(cx1, cy1, cx2, cy2, cx3, cy3)
            right_hand_angle = calculate_angle(cx4, cy4, cx5, cy5, cx6, cy6)

            mode = get_mode()
            if mode == 'normal':
                combine_counter = 0  # Reset combine counter
                if left_hand_angle <= down_thresh and not pushup_up_left:
                    pushup_up_left = True
                elif left_hand_angle >= up_thresh and pushup_up_left:
                    left_hand_counter += 1
                    pushup_up_left = False
                    speak(f'Left {left_hand_counter}')

                if right_hand_angle <= down_thresh and not pushup_up_right:
                    pushup_up_right = True
                elif right_hand_angle >= up_thresh and pushup_up_right:
                    right_hand_counter += 1
                    pushup_up_right = False
                    speak(f'Right {right_hand_counter}')

            elif mode == 'combine':
                left_hand_counter = 0  # Reset normal counters
                right_hand_counter = 0
                if left_hand_angle <= down_thresh and right_hand_angle <= down_thresh and not combine:
                    combine = True
                elif left_hand_angle >= up_thresh and right_hand_angle >= up_thresh and combine:
                    combine_counter += 1
                    combine = False
                    speak(f'Combine {combine_counter}')

        # Display counters
        if mode == 'normal':
            cvzone.putTextRect(frame, f'{int(left_hand_counter)}', (50, 60), 1, 2)
            cvzone.putTextRect(frame, f'{int(right_hand_counter)}', (50, 160), 1, 2)
        elif mode == 'combine':
            cvzone.putTextRect(frame, f'{int(combine_counter)}', (50, 60), 1, 2)

    # Show frame
    cv2.imshow("RGB", frame)

    # Check for exit
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

# Cleanup
speak(None)  # Stop speech thread
cap.release()
cv2.destroyAllWindows()