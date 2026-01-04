# config.py

# Model Path (Ensure this file is in your root or download logic is handled)
MODEL_PATH = 'yolo11n-pose.pt' 

# Counting Thresholds (Angles in degrees)
UP_THRESH = 150   # Arm fully extended
DOWN_THRESH = 90  # Arm curled up

# Colors (BGR Format for OpenCV)
COLOR_TEXT_MAIN = (0, 255, 0)   # Green
COLOR_TEXT_COUNT = (255, 0, 0)  # Blue
COLOR_LINE = (255, 255, 255)    # White
COLOR_JOINT = (0, 255, 255)     # Yellow