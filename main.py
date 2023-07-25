from src.video_face_detector import VideoFaceDetector


def detect_face_on_camera():
    # Detect faces on camera
    vfd = VideoFaceDetector(0)
    # vfd.stream()
    vfd.stream(vfd.highlight_face)
    # vfd.stream(vfd.highlight_face_landmarks)


def detect_face_on_video():
    # Detect faces on video
    vfd = VideoFaceDetector("data/video_test.mov")
    # vfd.stream()
    # vfd.stream(vfd.highlight_face)
    vfd.stream(vfd.highlight_face_landmarks)


if __name__ == '__main__':
    # detect_face_on_camera()
    detect_face_on_video()

