import time
from gpiozero import MotionSensor
from picamera import PiCamera
import tensorflow as tf
import numpy as np
import pygame
from twilio.rest import Client
import tflite_runtime.interpreter as tf.lite

# Initialize components
pir = MotionSensor(4)
camera = PiCamera()
pygame.mixer.init()

# Twilio configuration
account_sid = 'ACc6d4f16d4b820257be1cfbc15677be42'
auth_token = 'b036a6d8a144a811b369581046b3ac4e'
client = Client(account_sid, auth_token)
from_whatsapp_number = 'whatsapp:+16089108025'
to_whatsapp_number = 'whatsapp:+254716814392'

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/main/baboon_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to capture image
def capture_image():
    camera.capture('image.jpg')

# Function to load image and preprocess
def load_image():
    img = tf.io.read_file('image.jpg')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [input_details[0]['shape'][1], input_details[0]['shape'][2]])
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to detect baboon
def detect_baboon():
    img = load_image()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0] > 0.95  # Assuming the model outputs a probability

# Function to play sound
def play_sound(sound_file, duration):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.music.stop()

# Function to send WhatsApp message
def send_whatsapp_message(message):
    client.messages.create(body=message, from_=from_whatsapp_number, to=to_whatsapp_number)

# Main loop
while True:
    pir.wait_for_motion()
    print("Motion detected!")
    capture_image()
    if detect_baboon():
        print("Baboon detected with high confidence!")
        play_sound('/main/18-19khz.mp3', 60)
        time.sleep(30)
        if detect_baboon():
            play_sound('/main/20-21khz.mp3', 60)
            time.sleep(60)
            if detect_baboon():
                send_whatsapp_message("Baboon detected for over 90 seconds!")
                continue  # Restart the loop
