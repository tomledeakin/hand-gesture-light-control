# **Light Controlling Using Hand Gestures**

**Organization:** AI VIETNAM – COURSE 2024  

---

## **Introduction**

**Light Controlling Using Hand Gestures** is a project aimed at developing a system that allows users to control lights using hand gestures. The system leverages Google’s MediaPipe Gesture Recognizer combined with a deep learning model to identify predefined hand gestures and execute corresponding light control commands.

The project is divided into **three main steps**:
1. **Data Preparation (Step 0):** Using MediaPipe to collect and process hand gesture landmark data.
2. **Model Training (Step 1):** Building and training an MLP (Multi-Layer Perceptron) model to classify hand gestures.
3. **System Deployment (Step 2):** Real-time hand gesture recognition to control lights in simulation or real hardware.

---

## **Installation Guide**

### **System Requirements**
- **Operating System:** Windows/Linux/MacOS
- **Python Version:** 3.10
- **Webcam:** Required for real-time gesture recognition
- **Hardware (Optional):** 
  - 4-Relay module with Modbus RTU RS485 communication
  - USB-to-RS485 adapter for hardware communication
  - 3 light bulbs with sockets for real-world testing

### **Steps to Set Up**

#### **1. Environment Setup**
1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).
2. Create a Python 3.10 environment:
   ```bash
   conda create -n gesture_env python=3.10
   conda activate gesture_env
   ```
3. Install required libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

#### **2. Required Files**
Ensure the following files are in the project directory:
- `hand_gesture.yaml`: Contains the class definitions for gestures.
- `generate_landmark_data.py`: For collecting and preparing gesture data.
- `hand_gesture_recognition.ipynb`: Notebook for training the MLP model.
- `detect_simulation.py`: Script for real-time gesture recognition and light control.
---

## **Project Workflow**

### **Step 0: Data Preparation**
1. Use `generate_landmark_data.py` to record hand gesture data using a webcam:
   - Gestures are recorded based on predefined classes in `hand_gesture.yaml`.
   - Data is saved in CSV files (`landmark_train.csv`, `landmark_val.csv`, and `landmark_test.csv`).
2. Example configuration in `hand_gesture.yaml`:
   ```yaml
   gestures:
     0: "turn_off"
     1: "light1"
     2: "light2"
     3: "light3"
     4: "turn_on"
   ```

---

### **Step 1: Model Training**
1. Load the gesture dataset and train an MLP model using PyTorch.
2. The model uses:
   - **Input Layer:** 63 features (landmarks)
   - **Output Layer:** Number of gesture classes
   - **Hidden Layers:** 4 layers with ReLU, BatchNorm, and Dropout for optimization.
3. Save the trained model for deployment.

---

### **Step 2: Real-Time Gesture Recognition**
1. Run `detect_simulation.py`:
   - Use a webcam for real-time hand gesture recognition.
   - Predict gestures using the trained MLP model.
   - Simulate light control or interact with actual hardware (if available).
2. Modes:
   - **Simulation Mode:** Displays light states on the screen.
   - **Hardware Mode:** Controls physical lights via Modbus RTU RS485.

---

## **Features**
- **Real-Time Hand Gesture Recognition:** Processes gestures live via webcam.
- **Gesture-Based Light Control:** Turns lights ON/OFF based on gesture classes.
- **Hardware Compatibility (Optional):** Supports Modbus RTU RS485 for controlling real-world devices.
- **Customizable Gestures:** Easily add or modify gesture classes via `hand_gesture.yaml`.

---

## **Usage Instructions**

1. **Data Collection:**
   - Run `generate_landmark_data.py` to collect and save gesture data.
   - Assign each gesture to a specific key and record multiple samples.

2. **Model Training:**
   - Train the MLP model in `hand_gesture_recognition.ipynb`.
   - Save the trained model for real-time recognition.

3. **Real-Time Gesture Recognition:**
   - Run `detect_simulation.py`.
   - Use gestures to control light states in simulation or on hardware.

---

## **Acknowledgments**
This project utilizes:
- **Google MediaPipe Gesture Recognizer** for hand gesture detection.
- **PyTorch** for building and training the deep learning model.
- **OpenCV** for video and image processing.

---

## **Contributors**
- **AI VIETNAM – COURSE 2024**
- **Hai Nam LE**

For any questions or assistance, please contact tomledeakin@gmail.com.
