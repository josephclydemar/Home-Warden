# with_gui = 0
# try:
#     with_gui = int(input('Enter mode\n(0) without GUI, (1) with GUI: '))
# except Exception as e:
#     print(e)
#     quit()



import os
import time
import threading
import cv2 # type: ignore
import socketio # type: ignore
import requests # type: ignore
from requests_toolbelt import MultipartEncoder # type: ignore
from flask import Flask, render_template, Response, request # type: ignore
import RPi.GPIO as GPIO # type: ignore
# from picamera2 import Picamera2 # type: ignore
import numpy as np # type: ignore
from keras_facenet import FaceNet # type: ignore
import pickle
from picamera2.encoders import H264Encoder # type: ignore
from picamera2.outputs import CircularOutput # type: ignore
from picamera2 import Picamera2 # type: ignore


from constant_variables import REMOTE_SERVER_HOST, HTTP_REST_ENDPOINTS, TRIG_PIN, ECHO_PIN, LED_PIN, BUTTON_PIN, BUZZER_PIN, SERVO_PIN, PWM_FREQ

sio = socketio.Client()
is_socket_connected = False
pwm = None

picam_initialized = False
picam = None
is_recording = False
recording_duration = 5  # Global variable to store the recording duration
video_writer = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_net = FaceNet()
app = Flask(__name__, template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

last_request_time = 0
request_timeout = 10  # Timeout duration in seconds
distanceGlobal = 1000



# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins for Ultrasonic sensor
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUZZER_PIN, GPIO.OUT)


GPIO.setup(SERVO_PIN, GPIO.OUT)
# PWM = GPIO.PWM(SERVO_PIN, PWM_FREQ)




with open("./trained_models/data1.pkl", "rb") as file:
    database = pickle.load(file)


def open_door():
    # pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
    # pwm.start(0)
    # pwm.ChangeDutyCycle(4)
    # time.sleep(2) # Adjust time for your servo to reach 90 degrees

    # pwm.ChangeDutyCycle(8.5)
    # time.sleep(2)
    # GPIO.cleanup()
    # pwm.stop()
    # print("Called")
    global pwm
    if pwm is None:
        print("Setup here")
        pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
        pwm.start(0)

    print("Here")
    pwm.ChangeDutyCycle(4)
    time.sleep(2) # Adjust time for your servo to reach 90 degrees
    print("Here2")
    pwm.ChangeDutyCycle(8.5)
    # time.sleep(2)

def sound_buzzer(pin):
    for _ in range(10):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.05)


def get_distance(trig_pin, echo_pin):
	# Set trigger to HIGH for 10us
	GPIO.output(trig_pin, True)
	time.sleep(0.00001)
	GPIO.output(trig_pin, False)

	# Wait for echo pin to go high
	start_time = time.time()
	while GPIO.input(echo_pin) == 0:
		if time.time() - start_time > 0.1:
			# print("Timeout")
			return 6000
	pulse_start = time.time()

	# Wait for echo pin to go low
	while GPIO.input(echo_pin) == 1:
		if time.time() - start_time > 0.1:
			# print("Timeout")
			return 6000
	pulse_end = time.time()

	# Calculate distance (speed of sound: 343m/s or 34300cm/s)
	distance = (pulse_end - pulse_start) * 34300 / 2
	return distance



while not is_socket_connected:
    try:
        sio.connect(REMOTE_SERVER_HOST)
        print('socketio id:', sio.sid)
        is_socket_connected = True

        @sio.event
        def from_server_control_door(data):
            global PWM, BUZZER_PIN
            print('Open Door', data)
            sound_buzzer(BUZZER_PIN)
            open_door()
    except Exception as e:
        print(e)
        continue





def initialize_picamera():
    global picam_initialized, picam
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = 'RGB888'
    picam.preview_configuration.align()
    picam.configure('preview')
    picam.start()
    picam_initialized = True

        # threads = (threading.Thread(target=send_recorded_videos),)
        # for t in threads:
        #     t.start()
    
        

# Generator function to generate frames for streaming
def generate():
    global picam_initialized, picam, video_writer, is_recording, distanceGlobal
    if not picam_initialized:
        initialize_picamera()
    
    while True:
        distanceGlobal = get_distance(TRIG_PIN, ECHO_PIN)
        frame = picam.capture_array()
        cv2.putText(frame, f'distance: {round(distanceGlobal, 3)} cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if distanceGlobal < 50:
            # Define the endpoint URL
            endpoint_url = 'http://0.0.0.0:8080/start_recording'
            # Define the data to be sent in the POST request (duration)
            data = {'duration': recording_duration}
            try:
                # Send a POST request to start recording
                response = requests.post(endpoint_url, json=data)
                # Check if the request was successful
                if response.status_code == 200:
                    print('Recording started successfully.')
                    # time.sleep(3)
                    # response = requests.post('http://0.0.0.0:8080/stop_recording')
                    print(response.text)
                else:
                    print('Failed to start recording:', response.text)

            except Exception as e:
                print(e)

                

            # if not is_recording:
            #     video_writer = cv2.VideoWriter(os.path.join('videos', 'new', 'new_video.mkv'), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            #     is_recording = True

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9)
            if len(faces_rect) > 0:
                for x, y, w, h in faces_rect:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
                    face_signature = face_net.embeddings(np.expand_dims(face_img, axis=0))
                    
                    min_dist = 0.7
                    identity = 'Unknown'
                    
                    # Check distance to known faces in the database
                    for key, value in database.items():
                        dist = np.linalg.norm(value - face_signature)
                        if dist < min_dist:
                            min_dist = dist
                            identity = key
                    
                    # Draw rectangle around the face
                    color = (0, 0, 255) if identity == 'Unknown' else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    text = f"{identity} {dist:.2f}" if identity != "Unknown" else "Unknown"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if video_writer is not None:
                video_writer.write(frame)
        else:
            pass
            # if is_recording and video_writer is not None:
            #     video_writer.release()
            #     video_writer = None
            #     is_recording = False
            #     print('Created video...')
                # if os.path.exists(os.path.join('videos', 'new', 'new_video.mkv')):
                #     os.rename(os.path.join('videos', 'new', 'new_video.mkv'), os.path.join('videos', 'ready', f'{time.time_ns()}.mkv'))
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        byte_frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')



def send_recorded_videos():
    with requests.Session() as session:
        while True:
            if len(os.listdir(os.path.join('videos', 'ready'))) > 0:
                current_day_response = session.get(HTTP_REST_ENDPOINTS['day_records_v2'])
                current_day_json = current_day_response.json()
                for filename in os.listdir(os.path.join('videos', 'ready')):
                    file = open(os.path.join('videos', 'ready', filename), 'rb')
                    payload = MultipartEncoder(fields={
                        'recorded_video': (filename, file, 'video/h264'),
                        'day_record_id': current_day_json['_id']
                    })
                    response = session.post(HTTP_REST_ENDPOINTS['detections_v1'], data=payload, headers={'Content-Type': payload.content_type})
                    file.close()
                    print(response.json())
                    os.remove(os.path.join('videos', 'ready', filename))
                    # time.sleep(0.1)


def listen_for_gpio():
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            sio.emit('from_raspi_doorbell_press', 'The Doorbell has been pressed. Please check your camera.')
            sound_buzzer(BUZZER_PIN)


# def record_video(duration):
#     global video_writer
    # # Start recording
    # video_writer = cv2.VideoWriter('recorded_video2.mkv', cv2.VideoWriter_fourcc(*'H264'), 20.0, (640, 480))

    # # Record for the specified duration
    # start_time = time.time()
    # while time.time() - start_time < duration:
    #     frame = picam.capture_array()
    #     if video_writer is not None:  # Check if video_writer is initialized
    #         video_writer.write(frame)
    #     else:
    #         print("Video writer is not initialized!")
    #         break

    # # Release the video writer
    # if video_writer is not None:
    #     video_writer.release()
   







# if __name__ == '__main__':
#     threads = (threading.Thread(target=main), threading.Thread(target=send_recorded_videos))
#     for t in threads:
#         t.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global picam, is_recording, last_request_time, distanceGlobal
    if request.method == 'POST' and not is_recording and  time.time() - last_request_time >= 10:
    # if request.method == 'POST':
        # is_recording = True
        # Parse the duration from the request data
        encoder=H264Encoder()
        output=CircularOutput()
        # Start recording
        timestamp = str(int(time.time() * 1000))
        output_directory = os.path.join('videos', 'ready')
        output_file = os.path.join(output_directory, f'recorded_video_{timestamp}.h264')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        picam.start_recording(encoder, output)
        print("Called")
        try:
        # Start recording
            output.fileoutput=output_file
            output.start()

            # Continuous recording until distance exceeds 50
            
            # while time.time() - start_time2 < 1:  # Check distance every 0.1 seconds
            #     print("asd")
            # # while True:
            #     if distanceGlobal >= 50:
            #         break  # Exit the loop if distance exceeds 50
            #     else:
            #         start_time2 = time.time()
            while distanceGlobal < 50:
                time.sleep(0.1)
            #     time.sleep(0.1)
            # time.sleep(3)

            # Stop recording
            print("Recording Stopped, uploading")
            output.stop()
            last_request_time = time.time()
            is_recording = False
            return 'Recording completed.'
        
        
        except Exception as e:
            return f'Failed to start recording here: {str(e)}'

        return 'Recording completed.'
    else:
        return 'Recording not complete.'

# @app.route('/stop_recording', methods=['POST'])
# def stop_recording():
#     global video_writer
#     if request.method == 'POST':
#         # Stop recording (if currently recording)
#         if video_writer is not None:
#             video_writer.release()
#             video_writer = None
#             return 'Recording stopped successfully.'
#         else:
#             return 'No active recording to stop.'
#     else:
#         return 'Invalid request method.'


try:
    if __name__ == '__main__':
        threads = (
                # threading.Thread(target=send_recorded_videos),
                threading.Thread(target=listen_for_gpio),
            )
        for t in threads:
            t.start()


        app.run(host='0.0.0.0', port=8080, debug=False)
        
except KeyboardInterrupt:
    GPIO.cleanup()

























