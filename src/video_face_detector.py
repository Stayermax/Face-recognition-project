import cv2
import numpy as np
import face_recognition

class VideoFaceDetector:
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = {
        # BGR colors
        "red": (0,0,255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255)
    }
    face_landmarks_list = ['chin', 'left_eyebrow', 'right_eyebrow',
                           'nose_bridge', 'nose_tip', 'left_eye',
                           'right_eye', 'top_lip', 'bottom_lip',
                           'face']

    def __init__(self, input_source=0):
        """
        Input source can be port of camera or path to video
        :param input_source:
        """
        self.cap = cv2.VideoCapture(input_source)
        ret, image = self.cap.read()
        if not ret:
            raise Exception(f'Input source "{input_source}" is not accessible')
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.set_face_landmarks_random_colors()

    def stream(self, image_edit_function = lambda x: x):
        while True:
            ret, image = self.cap.read()
            cv2.imshow("Test", image_edit_function(image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def set_face_landmarks_random_colors(self):
        self.face_landmarks_colors = {}
        for face_landmark in VideoFaceDetector.face_landmarks_list:
            self.face_landmarks_colors[face_landmark] = \
                tuple(int(el) for el in tuple(np.random.choice(range(256), size=3)))


    @staticmethod
    def color_by_name(color: str) -> tuple:
        return VideoFaceDetector.colors.get(color, (255, 255, 255))


    @staticmethod
    def draw_rectangle(image: np.ndarray, face_cords: list, bgr_color: tuple):
        (top, right, bottom, left) = face_cords
        image = cv2.rectangle(image, (left, top), (right, bottom), bgr_color, 5)
        return cv2.putText(image, "Unknown", (left + 6, bottom - 6), VideoFaceDetector.font, 1.0, bgr_color, 2)

    @staticmethod
    def draw_contour(image: np.ndarray, cords_list: tuple, bgr_color: tuple):
        return cv2.polylines(img=image,
                             pts=np.int32([cords_list]),
                             isClosed=False,
                             color=bgr_color,
                             thickness=3)


    def detect_faces(self, image):
        rgb_small_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(cv2.resize(rgb_small_frame, (0, 0), fx=0.25, fy=0.25))
        face_locations = tuple([el*4 for el in fl] for fl in face_locations)
        return face_locations

    def highlight_face(self, image):
        cords_list = self.detect_faces(image)
        for face_cords in cords_list:
            image = self.draw_rectangle(image, face_cords, self.face_landmarks_colors["face"])
        return image

    def detect_faces_landmarks(self, image):
        cords_list = self.detect_faces(image)
        face_locations = face_recognition.face_landmarks(image, cords_list)
        return face_locations

    def highlight_face_landmarks(self, image):
        faces_cords_list = self.detect_faces_landmarks(image)
        for face_cords_list in faces_cords_list:
            for face_landmark in face_cords_list:
                image = VideoFaceDetector.draw_contour(image=image,
                                                       cords_list=face_cords_list[face_landmark],
                                                       bgr_color=self.face_landmarks_colors[face_landmark])
        return image
