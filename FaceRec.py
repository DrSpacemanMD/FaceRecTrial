from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import sys

# extract a single face from a given photograph
def extract_face(pixels, required_size=(224, 224)):
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	if (len(results)==0):
		return [],"No Face Found!"
	if (len(results)>1):
		return [],"Multiple Faces Found!"
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array,"Face Found"

# extract faces and calculate face embeddings for a list of photo files
def get_features(pixels): #Load list of images
	# extract faces
	faces,ReturnCode = extract_face(pixels)
	faces = [faces]
	if (ReturnCode != "Face Found"):
		return [],[],ReturnCode
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat,faces,ReturnCode

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return True, 1-score
	else:
		return False, 1-score



def SimpleWebcamTest():
	
	video_capture = cv2.VideoCapture(0)
	time.sleep(5)

	print("Taking first photo...")
	if not video_capture.isOpened():
		raise Exception("Could not open video device")
	ret, frame = video_capture.read()

	frameRGB = frame[:,:,::-1] # BGR => RGB
	features1,faces1 = get_features([frameRGB])
    
	
	time.sleep(2)
	print("Taking second photo...")
	ret, frame = video_capture.read()
	# Close device
	video_capture.release()
	frameRGB = frame[:,:,::-1] # BGR => RGB
	features2,faces2  = get_features([frameRGB])
	plt.imshow(frameRGB)
	plt.show()
	Result, score = is_match(features1[0], features2[0])


	f = plt.figure()
	f.add_subplot(1,2, 1)
	plt.imshow(faces1[0])
	f.add_subplot(1,2, 2)
	plt.imshow(faces2[0])

	if Result == True:
		print ("Face is a match!, Score: " + str(score))
	else:
		print ("Face is not a match!, Score: " + str(score))


	plt.show(block=True)






'''
# define filenames
filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg',
	'sharon_stone3.jpg', 'channing_tatum.jpg']

images = [pyplot.imread(filenames[0]),pyplot.imread(filenames[1]),pyplot.imread(filenames[2]),pyplot.imread(filenames[3])]

# get embeddings file filenames
embeddings,faces = get_features(images)

# define sharon stone
sharon_id = embeddings[0]
# verify known photos of sharon
print('Positive Tests')
is_match(embeddings[0], embeddings[1])
is_match(embeddings[0], embeddings[2])
# verify known photos of other people
print('Negative Tests')
is_match(embeddings[0], embeddings[3])
'''