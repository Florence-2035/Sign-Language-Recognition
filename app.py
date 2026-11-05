
import cv2
import mediapipe as mp
import time
import streamlit as st
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time Sign Language Recognition")
st.write("Recognize hand numbers and basic sign gestures.")

VALID_WORDS = ["BAD", "LAD", "DAY", "BEE", "LIVE", "YELL", "BALL"]


class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mode = "NUMBER"
        self.word = ""
        self.last_letter = ""
        self.last_time = 0
        self.lock = threading.Lock()

    def reset_word(self):
        with self.lock:
            self.word = ""
            self.last_letter = ""
            self.last_time = 0

    def recognize_letter(self, f):

        gestures = {
            (0, 0, 0, 0, 0): "A",
            (0, 1, 1, 1, 1): "B",
            (0, 1, 0, 0, 0): "D",
            (1, 0, 0, 0, 0): "E",
            (1, 1, 0, 0, 0): "L",
            (0, 1, 1, 0, 0): "V",
            (1, 0, 0, 0, 1): "Y",
            (0, 0, 0, 0, 1): "I"
        }

        return gestures.get(tuple(f), "")

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)

        total_fingers = 0
        gesture = ""

        first_hand_fingers = None

        if results.multi_hand_landmarks:

            for index, handLms in enumerate(results.multi_hand_landmarks):

                self.mp_draw.draw_landmarks(
                    img,
                    handLms,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = handLms.landmark

                hand_label = (
                    results.multi_handedness[index]
                    .classification[0].label
                )

                fingers = []

                # Thumb
                if hand_label == "Right":
                    fingers.append(
                        1 if landmarks[4].x < landmarks[3].x else 0
                    )
                else:
                    fingers.append(
                        1 if landmarks[4].x > landmarks[3].x else 0
                    )

                # Other fingers
                for tip in [8, 12, 16, 20]:
                    fingers.append(
                        1 if landmarks[tip].y <
                        landmarks[tip - 2].y else 0
                    )

                total_fingers += sum(fingers)

                if index == 0:
                    first_hand_fingers = fingers

        with self.lock:
            mode = self.mode

            if mode == "NUMBER":
                cv2.putText(
                    img,
                    f"Number: {total_fingers}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (0, 255, 0),
                    4
                )

            elif mode == "LETTER":

                if first_hand_fingers is not None:
                    gesture = self.recognize_letter(
                        first_hand_fingers
                    )

                current_time = time.time()

                if gesture:
                    if (
                        gesture != self.last_letter
                        and current_time - self.last_time > 2
                    ):
                        self.word += gesture
                        self.last_letter = gesture
                        self.last_time = current_time
                else:
                    self.last_letter = ""

                cv2.putText(
                    img,
                    f"Letter: {gesture}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (255, 0, 0),
                    4
                )

                cv2.putText(
                    img,
                    f"Word: {self.word}",
                    (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 0),
                    3
                )

                if self.word:
                    status = (
                        "VALID WORD"
                        if self.word in VALID_WORDS
                        else "NOT IN DICTIONARY"
                    )

                    color = (
                        (0, 255, 0)
                        if status == "VALID WORD"
                        else (0, 0, 255)
                    )

                    cv2.putText(
                        img,
                        status,
                        (50, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        3
                    )

            cv2.putText(
                img,
                f"Mode: {mode}",
                (50, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2
            )

        return frame.from_ndarray(img, format="bgr24")


st.subheader("Select Recognition Mode")

mode = st.radio(
    "Mode",
    ["NUMBER", "LETTER"],
    horizontal=True
)

ctx = webrtc_streamer(
    key="sign-language-camera",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

if ctx.video_processor:
    ctx.video_processor.mode = mode

    if st.button("Clear Word"):
        ctx.video_processor.reset_word()

st.caption(
    "Allow camera permission in your browser to start recognition."
)

st.info(
    "NUMBER mode: Count fingers on both hands. "
    "LETTER mode: Recognize supported gestures and form words."
)
