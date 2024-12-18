import RPi.GPIO as GPIO
import time
from picamera import PiCamera
from twilio.rest import Client
import tensorflow as tf
import numpy as np
import tflite_runtime.interpreter as tflite

# Setup GPIO pins
MOTION_SENSOR_PIN = 17
ULTRASONIC_SPEAKER_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTION_SENSOR_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_SPEAKER_PIN, GPIO.OUT)

# Initialize camera
camera = PiCamera()

# Load your TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='/main/baboon_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Twilio setup
account_sid = 'ACc6d4f16d4b820257be1cfbc15677be42'
auth_token = 'b036a6d8a144a811b369581046b3ac4e'
client = Client(account_sid, auth_token)

from_whatsapp_number = 'whatsapp:+16089108025'
to_whatsapp_number = 'whatsapp:+254716814392'

def capture_image():
    image_path = '/pi/image.jpg'
    camera.capture(image_path)
    return image_path

def analyze_image(image_path):
    # Preprocess image for the TFLite model
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.0  # Normalize

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0] > 0.5  # Assuming binary classification with monkey=1

def play_sound(sound_file, duration):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.music.stop()

def send_whatsapp_message(message):
    client.messages.create(body=message, from_=from_whatsapp_number, to=to_whatsapp_number)

    print(f"Alert sent: {message.sid}")

def main_loop():
    try:
        while True:
            if GPIO.input(MOTION_SENSOR_PIN):
                print("Motion detected!")

                # Capture and analyze image
                image_path = capture_image()
                if analyze_image(image_path):
                    print("Monkey detected!")

                    # Trigger sound at 19kHz for 60 seconds
                    play_sound('/main/18-19khz.mp3', 60)
                    time.sleep(30)

                    # Check motion again
                    if GPIO.input(MOTION_SENSOR_PIN):
                        image_path = capture_image()
                        if analyze_image(image_path):
                            print("Monkey still detected! Upshifting frequency.")

                            # Upshift sound to 21kHz for 90 seconds
                            play_sound('/main/20-21khz.mp3', 90)
                            time.sleep(30)

                            # Check motion again
                            if GPIO.input(MOTION_SENSOR_PIN):
                                image_path = capture_image()
                                if analyze_image(image_path):
                                    print("Monkey still detected! Sending distress message.")
                                    send_whatsapp_message("Baboon detected for over 90 seconds!")
                else:
                    print("No monkey detected.")

                # Wait before restarting the loop
                time.sleep(1)
            else:
                print("No motion detected.")
                time.sleep(0.5)  # Reduce CPU usage

    except KeyboardInterrupt:
        print("Exiting program.")
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()
