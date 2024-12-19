import RPi.GPIO as GPIO
import time
from picamera import PiCamera
import pygame
from twilio.rest import Client
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tflite_runtime.interpreter as tflite

# Setup GPIO pins
MOTION_SENSOR_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTION_SENSOR_PIN, GPIO.IN)

# Initialize components
camera = PiCamera()
pygame.mixer.init()

# Load the TensorFlow Lite model
MODEL_PATH = '/home/casper/Monkdetect/baboon_model.tflite'
LABELMAP_PATH = '/home/casper/Monkdetect/labelmap.txt'

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
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

# Load label map
with open(LABELMAP_PATH, 'r') as f:
    label_map = [line.strip() for line in f.readlines()]

def capture_image():
    image_path = '/home/casper/Monkdetect/image.jpg'
    camera.capture(image_path)
    return image_path

def analyze_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    input_shape = input_details[0]['shape']
    image_resized = image.resize((input_shape[1], input_shape[2]))
    
    # Preprocess image
    image_array = np.array(image_resized).astype('float32') / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    # Process detected objects
    baboon_detected = False
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = int(classes[i])
            label = label_map[class_id]
            if label == "baboon":
                baboon_detected = True
                draw_bounding_box(image, boxes[i], label, scores[i])

    # Save annotated image
    if baboon_detected:
        image.save('/home/casper/Monkdetect/annotated_image.jpg')
    return baboon_detected

def draw_bounding_box(image, box, label, score):
    """Draw bounding box and label on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    ymin, xmin, ymax, xmax = box

    # Scale box to image size
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)

    # Draw rectangle
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

    # Draw label
    label_text = f"{label}: {score:.2%}"
    font = ImageFont.load_default()
    text_width, text_height = draw.textsize(label_text, font=font)
    draw.rectangle([xmin, ymin - text_height - 4, xmin + text_width + 4, ymin], fill="red")
    draw.text((xmin + 2, ymin - text_height - 2), label_text, fill="white", font=font)

def play_sound(sound_file, duration):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.music.stop()

def send_whatsapp_message(message):
    try:
        message = client.messages.create(body=message, from_=from_whatsapp_number, to=to_whatsapp_number)
        print(f"Alert sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

def main_loop():
    try:
        while True:
            if GPIO.input(MOTION_SENSOR_PIN):
                print("Motion detected!")
                time.sleep(0.1)

                # Capture and analyze image
                image_path = capture_image()
                if analyze_image(image_path):
                    print("Baboon detected!")

                    # Play sound at 19kHz for 6 seconds
                    play_sound('/home/casper/Monkdetect/18-19khz.mp3', 6)

                    # If baboon still detected
                    if GPIO.input(MOTION_SENSOR_PIN):
                        image_path = capture_image()
                        if analyze_image(image_path):
                            print("Baboon still detected! Upshifting frequency.")

                            # Play sound at 21kHz for 9 seconds
                            play_sound('/home/casper/Monkdetect/20-21khz.mp3', 9)

                            # Send WhatsApp alert
                            send_whatsapp_message("Baboon detected! Distress alert sent.")
                else:
                    print("No baboon present.")
                time.sleep(1)  # Restart loop delay
            else:
                print("No motion detected.")
                time.sleep(0.5)  # Reduce CPU usage

    except KeyboardInterrupt:
        print("Exiting program.")
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()
