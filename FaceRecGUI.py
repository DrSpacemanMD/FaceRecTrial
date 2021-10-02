from tkinter import *
from PIL import ImageTk, Image
import cv2
import matplotlib.pyplot as plt
from FaceRec import *
import pickle
import glob
from dataclasses import dataclass
import tkinter.messagebox
from threading import Thread


class Face:
	def __init__(self, name, features, face):
		self.name = name
		self.features = features
		self.face = face
		


root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

#Load in the plcaeholder image
PlaceHolderImg = Image.open("PlaceHolder.jpg")
PlaceHolderImg = PlaceHolderImg.resize((224, 224), Image.ANTIALIAS)
PlaceHolder = ImageTk.PhotoImage(PlaceHolderImg)  


ImageFrame = Frame(root)
#Set up the last enrolled image
LastImage = Label(ImageFrame)
LastImage.imgtk = PlaceHolder
LastImage.configure(image=PlaceHolder)
LastImageText = Label(ImageFrame, text="last Image Taken")
LastImageText.grid(row=0,column=0)
LastImage.grid(row=1,column=0)

#set up the identifed image
IdentifedImage = Label(ImageFrame)
IdentifedImage.imgtk = PlaceHolder
IdentifedImage.configure(image=PlaceHolder)
IdText = Label(ImageFrame, text="Identfied Face")
IdText.grid(row=2,column=0)
IdentifedImage.grid(row=3,column=0)

ResultText = Label(ImageFrame, text="Click Identify to get result")
ResultText.grid(row=4,column=0)

ImageFrame.grid(row=0,column=1,sticky="N")

Controls = Frame(root)
EnrolNameFrame = Frame(Controls)
NameText = Label(EnrolNameFrame, text="Enrol Name")
NameText.grid(row=0,column=0,sticky="E")
NameEntry = Entry(EnrolNameFrame,width = 25)
NameEntry.insert(END, 'John')
NameEntry.grid(row=0,column=1,sticky="W")
Status = Label(EnrolNameFrame, text="Ready!")
Status.grid(row=0,column=2,sticky="E")

EnrolNameFrame.grid(row=0,column=0,sticky="W")




cap = cv2.VideoCapture(0)
def video_stream():
	_, frame = cap.read()
	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	img = Image.fromarray(cv2image)
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(1, video_stream)
	lmain.grid(row=0,column=0)

def GetWebCamFace():
	timeout = 10   # [seconds]
	timeout_start = time.time()
	while time.time() < timeout_start + timeout:
		ret, frame = cap.read()
		frameRGB = frame[:,:,::-1] # BGR => RGB
		features,faces,ReturnCode = get_features(frameRGB)
		
		if (ReturnCode=="Face Found"):
			break
		
	if  (ReturnCode != "Face Found"):
		messagebox.showinfo("Error", ReturnCode)
		return None, ReturnCode
	return Face(NameEntry.get(),features,Image.fromarray(faces[0])), ReturnCode

#Old Single thread code..
'''
def EnrolFace():
	faceobj = GetWebCamFace()
	faceTK = ImageTk.PhotoImage(faceobj.face)  
	EnrolImage.imgtk = faceTK
	EnrolImage.configure(image=faceTK)
	with open("Library/"+faceobj.name + ".pickle", 'wb') as file:
		pickle.dump(faceobj, file) 
'''	

def EnrolFace():
	Enrol = AsyncEnrol()
	Enrol.start()
	
class AsyncEnrol(Thread):
	def __init__(self):
		super().__init__()
	
	def run(self):
		Status["text"] = "Scanning!"
		faceobj,ReturnCode = GetWebCamFace()
		if (ReturnCode != "Face Found"):
			Status["text"] = "Ready!"
			return
		#faceTK = ImageTk.PhotoImage(faceobj.face)  
		#EnrolImage.imgtk = faceTK
		#EnrolImage.configure(image=faceTK)
		UpdateImage(faceobj)
		with open("Library/"+faceobj.name + ".pickle", 'wb') as file:
			pickle.dump(faceobj, file) 
		Status["text"] = "Ready!"
			
def UpdateImage(faceobj):	
	faceTK = ImageTk.PhotoImage(faceobj.face)  
	LastImage.imgtk = faceTK
	LastImage.configure(image=faceTK)
#Old Single thread code..		
'''
def IdentifyFace():	
	faceobj = GetWebCamFace()
	files = glob.glob("Library/*.pickle")
	
	BestScore = -1
	BestMatch = None
	for file in files:
		with open(file, 'rb') as loadedfile:
			LoadedFace = pickle.load(loadedfile)
			Result, score = is_match(faceobj.features, LoadedFace.features)
			if (Result==True):
				if (score > BestScore):
					BestScore=score
					BestMatch=LoadedFace
					
	if (BestMatch!=None):
		faceTK = ImageTk.PhotoImage(faceobj.face)  
		IdentifedImage.imgtk = faceTK
		IdentifedImage.configure(image=faceTK)
		ResultText['text'] ="Identified as " + faceobj.name +" " + str(round(BestScore*100,2))+"%"
	else:
		ResultText['text'] ="No positive match"
'''			

def IdentifyFace():				
	Identify = AsyncIdentify()
	Identify.start()

class AsyncIdentify(Thread):
	def __init__(self):
		super().__init__()
	
	def run(self):
		Status["text"] = "Scanning!"
		faceobj,ReturnCode = GetWebCamFace()
		if (ReturnCode != "Face Found"):
			Status["text"] = "Ready!"
			return
		UpdateImage(faceobj)
		files = glob.glob("Library/*.pickle")
		
		BestScore = -1
		BestMatch = None
		for file in files:
			with open(file, 'rb') as loadedfile:
				LoadedFace = pickle.load(loadedfile)
				Result, score = is_match(faceobj.features, LoadedFace.features)
				if (Result==True):
					if (score > BestScore):
						BestScore=score
						BestMatch=LoadedFace
						
		if (BestMatch!=None):
			faceTK = ImageTk.PhotoImage(BestMatch.face)  
			IdentifedImage.imgtk = faceTK
			IdentifedImage.configure(image=faceTK)
			ResultText['text'] ="Identified as " + BestMatch.name +" " + str(round(BestScore*100,2))+"%"
		else:
			ResultText['text'] ="No positive match"
			
		Status["text"] = "Ready!"
	
	
EnrolButton=Button(Controls, text="Enrol",height = 1,width = 30,command = EnrolFace)
EnrolButton.grid(row=2,column=0)

IdentifyButton=Button(Controls, text="Identify",height = 1,width = 30, command = IdentifyFace)
IdentifyButton.grid(row=2,column=1)
Controls.grid(row=1,column=0,sticky="W")


video_stream()
root.mainloop()