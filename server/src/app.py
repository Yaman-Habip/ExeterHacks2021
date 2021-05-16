# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import server as s

# Initialize the Flask app
app = Flask(__name__)


def gen_frames(server):

    try:
        while True:
            server.init_image_len()
            if not server.image_len:
                break
            server.get_frame()

            print('1')

            img_path = "C:/Users/jryan/PycharmProjects/python video stream/test_yolo3.jpg"
            # success = cv2.imread(img_path)  # read the camera frame
            frame = cv2.imread(img_path)  # read the camera frame

            print('2')
            # print('frame:', frame)

            if frame is None:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            print('3')
    finally:
        server.close_stream()


@app.route('/')
def sample():
    return render_template('sample.html')


@app.route('/video_stream')
def video_stream():
    server = s.Server()
    # return render_template('video_stream.html')
    return Response(gen_frames(server), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_page')
def video_page():
    return render_template('video_page.html')


app.run(host='0.0.0.0', port=5000)
