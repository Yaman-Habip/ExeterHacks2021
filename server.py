import io
import socket
import struct
from PIL import Image
import matplotlib.pyplot as pl

import image_processing.ImageDetection as id

# for opencv
import numpy as np


class Server():
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.1.38', 8000))  # ADD IP HERE
        self.server_socket.listen(0)

        # Accept a single connection and make a file-like object out of it
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # image that will be painted to the screen
        self.img = None

        self.image_len = None
        self.im = None

    # when used, put inside try block
    def init_image_len(self):
        self.image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

    def get_frame(self):
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        # print("type a:", type(connection.read(image_len)))
        image_stream.write(self.connection.read(self.image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        print(type(image_stream))
        image = Image.open(image_stream)

        byteImg = Image.open(image_stream)
        print("ASASD", type(byteImg))

        path_name = "C:/Users/jryan/PycharmProjects/python video stream/test.jpg"
        byteImg.save(path_name, 'JPEG')

        # im = Image.frombuffer("I;16", (5, 10), connection.read(image_len), "raw", "I;12")
        # print(type(im))

        # show yolo-ed image, if it doesn't exist then just show plain image
        print('Image type:', type(image))

        # matplotlib GUI for debugging
        # if self.img is None:
        #     self.img = image  # pl.imshow(image)  # image
        # else:
        #     img_path = "C:/Users/jryan/PycharmProjects/python video stream/test_yolo3.jpg"
        #     self.im = Image.open(img_path)
        #     self.img.set_data(self.im)

        # pl.pause(0.01)
        # pl.draw()

        print('Image is %dx%d' % image.size)

        image.verify()
        print('Image type:', type(image))
        id.go(image)  # run yolo program on the image
        print('Image is verified')

    # closes the stream (in 'finally' block)
    def close_stream(self):
        self.connection.close()
        self.server_socket.close()

# def run():
#     server_socket = socket.socket()
#     server_socket.bind(('192.168.1.38', 8000))  # ADD IP HERE
#     server_socket.listen(0)
#
#     # Accept a single connection and make a file-like object out of it
#     connection = server_socket.accept()[0].makefile('rb')


# def get_stream():
#     try:
#         img = None
#         # Read the length of the image as a 32-bit unsigned int. If the
#         # length is zero, quit the loop
#         image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
#         if not image_len:
#             break
#         # Construct a stream to hold the image data and read the image
#         # data from the connection
#         image_stream = io.BytesIO()
#         # print("type a:", type(connection.read(image_len)))
#         image_stream.write(connection.read(image_len))
#         # Rewind the stream, open it as an image with PIL and do some
#         # processing on it
#         image_stream.seek(0)
#         print(type(image_stream))
#         image = Image.open(image_stream)
#
#         byteImg = Image.open(image_stream)
#         print("ASASD", type(byteImg))
#         byteImg.save('test.jpg', 'JPEG')
#
#         # im = Image.frombuffer("I;16", (5, 10), connection.read(image_len), "raw", "I;12")
#         # print(type(im))
#
#         print('Image type:', type(image))
#         if img is None:
#             img = pl.imshow(image)
#         else:
#             im = Image.open("test_yolo3.jpg")
#             img.set_data(im)
#
#         pl.pause(0.01)
#         pl.draw()
#
#         print('Image is %dx%d' % image.size)
#
#         image.verify()
#         print('Image type:', type(image))
#         id.go(image)  # run yolo program on the image
#         print('Image is verified')
#     finally:
#         connection.close()
#         server_socket.close()
#
#
# try:
#     img = None
#     while True:
#         # Read the length of the image as a 32-bit unsigned int. If the
#         # length is zero, quit the loop
#         image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
#         if not image_len:
#             break
#         # Construct a stream to hold the image data and read the image
#         # data from the connection
#         image_stream = io.BytesIO()
#         # print("type a:", type(connection.read(image_len)))
#         image_stream.write(connection.read(image_len))
#         # Rewind the stream, open it as an image with PIL and do some
#         # processing on it
#         image_stream.seek(0)
#         print(type(image_stream))
#         image = Image.open(image_stream)
#
#         byteImg = Image.open(image_stream)
#         print("ASASD", type(byteImg))
#         byteImg.save('test.jpg', 'JPEG')
#
#         # im = Image.frombuffer("I;16", (5, 10), connection.read(image_len), "raw", "I;12")
#         # print(type(im))
#
#         print('Image type:', type(image))
#         if img is None:
#             img = pl.imshow(image)
#         else:
#             im = Image.open("test_yolo3.jpg")
#             img.set_data(im)
#
#         pl.pause(0.01)
#         pl.draw()
#
#         print('Image is %dx%d' % image.size)
#
#         image.verify()
#         print('Image type:', type(image))
#         id.go(image)  # run yolo program on the image
#         print('Image is verified')
# finally:
#     connection.close()
#     server_socket.close()
