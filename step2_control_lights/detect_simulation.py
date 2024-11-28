import os
import cv2
import time
import yaml
import torch
import numpy as np
from torch import nn
import mediapipe as mp

# Mock class to replace ModbusMaster for simulation
class MockModbusMaster:
    def switch_actuator_1(self, state):
        print(f"Mock: Switching actuator 1 to {'ON' if state else 'OFF'}")

    def switch_actuator_2(self, state):
        print(f"Mock: Switching actuator 2 to {'ON' if state else 'OFF'}")

    def switch_actuator_3(self, state):
        print(f"Mock: Switching actuator 3 to {'ON' if state else 'OFF'}")


class HandLandmarksDetector:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)

    def detectHand(self, frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
                hands.append(hand)
        return hands, annotated_image


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(list_label)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, x, threshold=0.9):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob, dim=1)
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)

    def predict_with_known_class(self, x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob, dim=1)


def label_dict_from_config_file(relative_path):
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag


class LightGesture:
    def __init__(self, model_path, device=False):
        self.device = device
        self.height = 720
        self.width = 1280

        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file("hand_gesture.yaml")
        self.classifier = NeuralNetwork()

        # Load the model
        self.classifier.load_state_dict(torch.load(
            model_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ))
        print("Model loaded successfully.")
        self.classifier.eval()

        # Replace ModbusMaster with mock class for simulation
        if self.device:
            from controller import ModbusMaster
            self.controller = ModbusMaster()
        else:
            self.controller = MockModbusMaster()

        self.light1 = False
        self.light2 = False
        self.light3 = False

        # Load and resize the images of lights (on/off)
        self.light1_on_img = self.load_and_resize_image('light1_on.png', (100, 100))
        self.light1_off_img = self.load_and_resize_image('light1_off.png', (100, 100))
        self.light2_on_img = self.load_and_resize_image('light2_on.png', (100, 100))
        self.light2_off_img = self.load_and_resize_image('light2_off.png', (100, 100))
        self.light3_on_img = self.load_and_resize_image('light3_on.png', (100, 100))
        self.light3_off_img = self.load_and_resize_image('light3_off.png', (100, 100))

        # Positions for the lights on the screen
        self.light1_pos = (50, 50)
        self.light2_pos = (200, 50)
        self.light3_pos = (350, 50)

    def load_and_resize_image(self, path, size):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Image at {path} could not be loaded.")
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def overlay_image_alpha(self, img, img_overlay, pos, alpha_mask):
        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis] / 255.0

        # Perform blending
        img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)
        while cam.isOpened():
            _, frame = cam.read()

            hand, img = self.detector.detectHand(frame)
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(np.array(hand[0], dtype=np.float32).flatten()).unsqueeze(0)
                    class_number = self.classifier.predict(hand_landmark).item()
                    if class_number != -1:
                        self.status_text = self.signs[class_number]

                        if self.status_text == "light1":
                            if not self.light1:
                                print("Turning light 1 ON")
                                self.light1 = True
                                self.controller.switch_actuator_1(True)
                        elif self.status_text == "light2":
                            if not self.light2:
                                print("Turning light 2 ON")
                                self.light2 = True
                                self.controller.switch_actuator_2(True)
                        elif self.status_text == "light3":
                            if not self.light3:
                                print("Turning light 3 ON")
                                self.light3 = True
                                self.controller.switch_actuator_3(True)
                        elif self.status_text == "turn_on":
                            self.light1 = self.light2 = self.light3 = True
                            print("Turning all lights ON")
                            self.controller.switch_actuator_1(True)
                            self.controller.switch_actuator_2(True)
                            self.controller.switch_actuator_3(True)
                        elif self.status_text == "turn_off":
                            self.light1 = self.light2 = self.light3 = False
                            print("Turning all lights OFF")
                            self.controller.switch_actuator_1(False)
                            self.controller.switch_actuator_2(False)
                            self.controller.switch_actuator_3(False)
                    else:
                        self.status_text = "undefined command"

            else:
                self.status_text = None

            # Display status text
            cv2.putText(img, self.status_text or "No Command", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Overlay the light images
            if self.light1:
                light1_img = self.light1_on_img
            else:
                light1_img = self.light1_off_img

            if self.light2:
                light2_img = self.light2_on_img
            else:
                light2_img = self.light2_off_img

            if self.light3:
                light3_img = self.light3_on_img
            else:
                light3_img = self.light3_off_img

            # Overlay images onto the frame
            self.overlay_image_alpha(img, light1_img[:, :, 0:3], self.light1_pos, light1_img[:, :, 3])
            self.overlay_image_alpha(img, light2_img[:, :, 0:3], self.light2_pos, light2_img[:, :, 3])
            self.overlay_image_alpha(img, light3_img[:, :, 0:3], self.light3_pos, light3_img[:, :, 3])

            # Show the image
            cv2.imshow("Simulation", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "../models/model_28-11 10_52_NeuralNetwork_best"
    try:
        light = LightGesture(model_path, device=False)  # Set device=False for simulation
        light.run()
    except Exception as e:
        print(f"Error: {e}")
