# Real-Time Sign Language Recognition Using Computer Vision

A Python-based computer vision project that detects hand landmarks through a webcam, counts raised fingers, recognizes selected hand gestures as letters, and checks formed letter sequences against a predefined word list.

## Features

- **Real-Time Hand Detection:** Detects and displays hand landmarks using MediaPipe.
- **Finger Counting:** Counts raised fingers across up to two detected hands.
- **Gesture Recognition:** Uses predefined finger patterns to recognize selected letters: A, B, D, E, L, V, Y and I.
- **Basic Word Formation:** Builds a sequence of recognized letters.
- **Word Validation:** Checks the sequence against a small predefined list of words.
- **Interactive Controls:** Switch between number and letter modes using keyboard shortcuts.

## Technologies Used

- Python
- OpenCV
- MediaPipe

## How It Works

1. Captures video from a webcam using OpenCV.
2. Converts video frames into the format required by MediaPipe.
3. Detects hand landmarks and draws them on the video feed.
4. Estimates raised fingers using landmark positions.
5. Maps selected finger patterns to letters in letter mode.
6. Checks the resulting letter sequence against predefined words.

## Getting Started

### Prerequisites

- Python installed on your computer
- A working webcam
- Git (optional, for cloning the repository)

### Installation

Clone the repository:

```bash
git clone https://github.com/Florence-2035/Sign-Language-Recognition.git
```

Open the project folder:

```bash
cd Sign-Language-Recognition
```

Install the required libraries:

```bash
pip install opencv-python mediapipe
```

Run the application:

```bash
python app.py
```

Allow webcam access if prompted.

## Keyboard Controls

| Key | Action |
|---|---|
| `n` | Switch to number mode |
| `l` | Switch to letter mode |
| `c` | Clear the current word |
| `q` | Quit the application |

## Current Limitations

- Only selected finger patterns are recognized.
- Letter recognition uses manually defined rules rather than a trained sign-language recognition model.
- Word checking uses a small predefined word list.
- Recognition may be affected by hand orientation, camera position, and lighting.
- The project does not currently translate complete sign-language sentences.

## Future Improvements

- Improve gesture recognition accuracy.
- Add support for more letters and gestures.
- Expand word formation and validation.
- Develop a web-based interface.
- Explore machine learning models for more advanced sign-language recognition.

## Author

**Florence Gaddam**

Computer Science and Engineering Student

[GitHub Profile](https://github.com/Florence-2035)

