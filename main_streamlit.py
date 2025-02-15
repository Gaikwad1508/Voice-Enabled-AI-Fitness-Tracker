import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import cvzone
import speech_recognition as sr
import time
import streamlit as st

# Initialize the YOLO model and video capture
model = YOLO('yolo11n-pose.pt')

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

engine = pyttsx3.init()  # Text-to-speech
voices = engine.getProperty('voices')    # Getting details of current voice
engine.setProperty('rate', 150)         # Setting up new voice rate
engine.setProperty('voice', voices[1].id)  # Setting female voice
speech_queue = queue.Queue()  # Speech queue

mode = None

# Listen for voice commands
def listen_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        if mode == 'stop':
            time.sleep(0.1)
            print('Listening stopped')
            break
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print('Listening...')
                audio = recognizer.listen(source)
                commands = recognizer.recognize_google(audio).lower()
                print(commands)
                if 'normal' in commands:
                    speak('Normal mode started')
                    set_mode('normal')
                elif 'combine' in commands:
                    speak('Combine mode started')
                    set_mode('combine')
                elif 'stop' in commands:
                    speak('Take care and have a nice day')
                    time.sleep(3)  # Ensure "Take care" is fully audible
                    set_mode('stop')
                    
        except sr.UnknownValueError:
            print('Could not understand the audio')

# Set mode
def set_mode(new_mode):
    global mode
    mode = new_mode

# Add text to speech
def speak(text):
    speech_queue.put(text)

# Worker thread for speech
def worker_speak():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit the thread gracefully
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Calculate the angle between three points (arms)
def angle(px1, py1, px2, py2, px3, py3):
    a = np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
    b = np.sqrt((px3 - px2) ** 2 + (py3 - py2) ** 2)
    c = np.sqrt((px3 - px1) ** 2 + (py3 - py1) ** 2)
    angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return np.degrees(angle)

# Start threads
threading.Thread(target=worker_speak, daemon=True).start()
threading.Thread(target=listen_commands, daemon=True).start()

# Streamlit setup
st.title("Voice Enabled AI Bicep Curl Tracker")
video_placeholder = st.empty()
mode_text_placeholder = st.empty()

# Main loop for video capture and processing
cap = cv2.VideoCapture(0)
while True:
    if mode == 'stop':  # Break if mode is 'stop'
        break

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 2 != 0:
        continue

    # Update mode text in Streamlit
    if mode == 'normal':
        mode_text_placeholder.markdown("### Single Hand Reps")
    elif mode == 'combine':
        mode_text_placeholder.markdown("### Double Hand Reps")

    # Make predictions
    result = model.track(frame)
    if result[0].boxes is not None and result[0].boxes.id is not None:
        keypoints = result[0].keypoints.xy.cpu().numpy()
        for keypoint in keypoints:
            if len(keypoint) > 0:
                for i, point in enumerate(keypoint):
                    cx, cy = int(point[0]), int(point[1])
                    cvzone.putTextRect(frame, f'{i}', (cx, cy), 1, 2)
                if mode and len(keypoint) > 8:
                    # Left hand
                    cx1, cy1 = int(keypoint[5][0]), int(keypoint[5][1])
                    cx2, cy2 = int(keypoint[7][0]), int(keypoint[7][1])
                    cx3, cy3 = int(keypoint[9][0]), int(keypoint[9][1])

                    # Right hand
                    cx4, cy4 = int(keypoint[6][0]), int(keypoint[6][1])
                    cx5, cy5 = int(keypoint[8][0]), int(keypoint[8][1])
                    cx6, cy6 = int(keypoint[10][0]), int(keypoint[10][1])

                    left_hand_angle = angle(cx1, cy1, cx2, cy2, cx3, cy3)
                    right_hand_angle = angle(cx4, cy4, cx5, cy5, cx6, cy6)

                    if mode == 'normal':
                        combine_counter = 0
                        # Left hand push-up counter
                        if left_hand_angle <= down_thresh and not pushup_up_left:
                            pushup_up_left = True
                        elif left_hand_angle >= up_thresh and pushup_up_left:
                            left_hand_counter += 1
                            pushup_up_left = False
                            speak(f'Left {left_hand_counter}')

                        # Right hand push-up counter
                        if right_hand_angle <= down_thresh and not pushup_up_right:
                            pushup_up_right = True
                        elif right_hand_angle >= up_thresh and pushup_up_right:
                            right_hand_counter += 1
                            pushup_up_right = False
                            speak(f'Right {right_hand_counter}')

                    elif mode == 'combine':
                        left_hand_counter = 0
                        right_hand_counter = 0
                        # Combine push-up counter
                        if left_hand_angle <= down_thresh and right_hand_angle <= down_thresh and not combine:
                            combine = True
                        elif left_hand_angle >= up_thresh and right_hand_angle >= up_thresh and combine:
                            combine_counter += 1
                            combine = False
                            speak(f'Combine {combine_counter}')

        # Display the counter
        if mode == 'normal':
            cvzone.putTextRect(frame, f'Left: {int(left_hand_counter)}', (50, 60), 1, 2)
            cvzone.putTextRect(frame, f'Right: {int(right_hand_counter)}', (50, 160), 1, 2)
        elif mode == 'combine':
            cvzone.putTextRect(frame, f'Combine: {int(combine_counter)}', (50, 60), 1, 2)

    # Display the frame in Streamlit
    video_placeholder.image(frame, channels="BGR")

# Cleanup
cap.release()
