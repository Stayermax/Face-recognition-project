from src.video_face_detector import VideoFaceDetector

# Show options:
BOX = "box"
LANDMARKS = "landmarks"
NOTHING = "nothing"

def detect_face_on_camera(what_to_show: str = BOX):
    # Detect faces on camera
    vfd = VideoFaceDetector(0)
    if what_to_show == BOX:
        vfd.stream(vfd.highlight_face)
    elif what_to_show == LANDMARKS:
        vfd.stream(vfd.highlight_face_landmarks)
    elif what_to_show == NOTHING:
        vfd.stream()

def detect_face_on_video(path_to_video: str, what_to_show: str = BOX):
    # Detect faces on video
    vfd = VideoFaceDetector(path_to_video)
    if what_to_show == BOX:
        vfd.stream(vfd.highlight_face)
    elif what_to_show == LANDMARKS:
        vfd.stream(vfd.highlight_face_landmarks)
    elif what_to_show == NOTHING:
        vfd.stream()

if __name__ == '__main__':
    detect_face_on_camera(what_to_show=LANDMARKS)
    # detect_face_on_video("data/video.mp4")

