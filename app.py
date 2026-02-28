import cv2
import mediapipe as mp
import time

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

mode = "NUMBER"
word = ""
last_letter = ""
last_time = time.time()

# Simple dictionary
valid_words = ["BAD", "LAD", "DAY", "BEE", "LIVE", "YELL", "BALL"]

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0
    gesture = ""

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_index, handLms in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[hand_index].classification[0].label
            landmarks = handLms.landmark
            fingers = []

            # ---- THUMB (Correct for left & right) ----
            if hand_label == "Right":
                fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
            else:
                fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

            # ---- OTHER FINGERS ----
            tip_ids = [8, 12, 16, 20]
            for tip in tip_ids:
                fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

            total_fingers += sum(fingers)

        # ================= NUMBER MODE =================
        if mode == "NUMBER":
            cv2.putText(img, f'Number: {total_fingers}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

        # ================= LETTER MODE =================
        elif mode == "LETTER":

            # Use first detected hand for letter recognition
            f = fingers

            if f == [0,0,0,0,0]:
                gesture = "A"

            elif f == [0,1,1,1,1]:
                gesture = "B"

            elif f == [0,1,0,0,0]:
                gesture = "D"

            elif f == [1,0,0,0,0]:
                gesture = "E"

            elif f == [1,1,0,0,0]:
                gesture = "L"

            elif f == [0,1,1,0,0]:
                gesture = "V"

            elif f == [1,0,0,0,1]:
                gesture = "Y"

            elif f == [0,0,0,0,1]:
                gesture = "I"

            else:
                gesture = ""

            current_time = time.time()

            if gesture != "" and gesture != last_letter:
                if current_time - last_time > 2:
                    word += gesture
                    last_letter = gesture
                    last_time = current_time

            cv2.putText(img, f'Letter: {gesture}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4)

    # Show Word
    if mode == "LETTER":
        cv2.putText(img, f'Word: {word}', (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        # Word checking
        if len(word) > 0:
            if word in valid_words:
                cv2.putText(img, "VALID WORD", (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(img, "NOT IN DICTIONARY", (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show Mode
    cv2.putText(img, f'Mode: {mode}', (50, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("Multi-Hand Counter + Gesture Recognition", img)

    key = cv2.waitKey(1) & 0xFF

    # Switch modes
    if key == ord('n'):
        mode = "NUMBER"

    if key == ord('l'):
        mode = "LETTER"

    # Clear word
    if key == ord('c'):
        word = ""

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()