import cv2
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase
import config
from src.utils import calculate_angle

class BicepCurlProcessor(VideoTransformerBase):
    def __init__(self, mode):
        self.mode = mode
        try:
            self.model = YOLO(config.MODEL_PATH)
        except:
            self.model = None

        # Counters
        self.count_left = 0
        self.count_right = 0
        self.count_combine = 0
        
        # States (0: Down/Relaxed, 1: Up/Curled)
        self.state_left = 0 
        self.state_right = 0
        self.state_combine = 0

        self.frame_counter = 0

    def draw_status(self, img, text, pos, color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draws text with a background box"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + 5), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, color, thickness)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Frame Skipping (Performance)
        self.frame_counter += 1
        if self.frame_counter % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 2. Run Inference
        results = self.model.track(img, persist=True, verbose=False)
        
        # 3. "Focus Mode" - Find the Largest Person
        # We look for the bounding box with the largest area
        max_area = 0
        main_person_idx = -1

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                # Calculate Area: (x2 - x1) * (y2 - y1)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    main_person_idx = i

        # 4. Process ONLY the Main Person
        if main_person_idx != -1 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            # Select keypoints for the main person only
            kps = keypoints[main_person_idx]

            if len(kps) > 10:
                l_s, l_e, l_w = kps[5], kps[7], kps[9]
                r_s, r_e, r_w = kps[6], kps[8], kps[10]

                angle_left = calculate_angle(l_s, l_e, l_w)
                angle_right = calculate_angle(r_s, r_e, r_w)

                # --- COUNTING LOGIC ---
                if "Normal" in self.mode:
                    # Left
                    if angle_left > config.UP_THRESH: self.state_left = 1
                    if angle_left < config.DOWN_THRESH and self.state_left == 1:
                        self.state_left = 0
                        self.count_left += 1

                    # Right
                    if angle_right > config.UP_THRESH: self.state_right = 1
                    if angle_right < config.DOWN_THRESH and self.state_right == 1:
                        self.state_right = 0
                        self.count_right += 1

                elif "Combine" in self.mode:
                    if angle_left > config.UP_THRESH and angle_right > config.UP_THRESH:
                        self.state_combine = 1
                    if angle_left < config.DOWN_THRESH and angle_right < config.DOWN_THRESH and self.state_combine == 1:
                        self.state_combine = 0
                        self.count_combine += 1

                # Draw Skeleton (Only for main person)
                for p1, p2 in [(l_s, l_e), (l_e, l_w), (r_s, r_e), (r_e, r_w)]:
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), config.COLOR_LINE, 3)
                for p in [l_s, l_e, l_w, r_s, r_e, r_w]:
                    cv2.circle(img, (int(p[0]), int(p[1])), 6, config.COLOR_JOINT, -1)

        # 5. Draw UI (OUTSIDE the loop so it never flickers)
        if "Normal" in self.mode:
            self.draw_status(img, f'Left: {self.count_left}', (30, 50), bg_color=(0,0,0))
            self.draw_status(img, f'Right: {self.count_right}', (30, 100), bg_color=(0,0,0))
        elif "Combine" in self.mode:
            self.draw_status(img, f'Combine: {self.count_combine}', (30, 50), bg_color=(0,0,0))

        return av.VideoFrame.from_ndarray(img, format="bgr24")