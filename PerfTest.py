from FaceRec import *
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
def GetScores():
	path = "C:/Users/Johnt/Documents/GitHub/FaceRecTrial/lfw/lfw"
	file = 'pairs.txt'
	num_lines = sum(1 for line in open(file))-1
	f = open(file)
	MatchScores = []
	MisMatchScores = []
	
	n=0
	for line in f:
		if len(line.split()) == 3: #Match
			person1 = line.split()[0]
			photonumber1 = line.split()[1]
			photonumber2 = line.split()[2]	
			
			Photo1Path = path + "/" + person1 + "/" + person1 + "_" + photonumber1.zfill(4)+".jpg"
			Photo2Path = path + "/" + person1 + "/" + person1 + "_" + photonumber2.zfill(4)+".jpg"
			
			Image1 = pyplot.imread(Photo1Path)
			Image2 = pyplot.imread(Photo2Path)
			
			features1,_,ReturnCode1 = get_features(Image1)
			features2,_,ReturnCode2 = get_features(Image2)
			
			if (ReturnCode1 == "Face Found" and ReturnCode2 == "Face Found"):
				_,Score = is_match(features1,features2,thresh=0)
				MatchScores.append(Score)
				
			n+=1
			print (str(n)+"/"+str(num_lines))
			
			
			
		if len(line.split()) == 4: #MisMatch
			person1 = line.split()[0]
			photonumber1 = line.split()[1]
			person2 = line.split()[2]
			photonumber2 = line.split()[3]	
			
			Photo1Path = path + "/" + person1 + "/" + person1 + "_" + photonumber1.zfill(4)+".jpg"
			Photo2Path = path + "/" + person2 + "/" + person2 + "_" + photonumber2.zfill(4)+".jpg"
			
			Image1 = pyplot.imread(Photo1Path)
			Image2 = pyplot.imread(Photo2Path)
			
			features1,_,ReturnCode1 = get_features(Image1)
			features2,_,ReturnCode2 = get_features(Image2)
			
			if (ReturnCode1 == "Face Found" and ReturnCode2 == "Face Found"):
				_,Score = is_match(features1,features2,thresh=0)
				MisMatchScores.append(Score)
	
			n+=1
			print (str(n)+"/"+str(num_lines))
	 
	np.save("MatchScores.npy", MatchScores)
	np.save("MisMatchScores.npy", MisMatchScores)


	
	
def GetAccuracyKFolds(FoldNumbers):
	
	MatchScores = list(np.load("MatchScores.npy"))
	MisMatchScores = list(np.load("MisMatchScores.npy"))
	MatchScores_folds = np.array_split(MatchScores, FoldNumbers)
	MisMatchScores_folds = np.array_split(MisMatchScores, FoldNumbers)
	
	AvgTA=0
	AvgFA=0
	AvgTR=0
	AvgFR=0
	AvgAcc=0

	for TestFold in range(0,FoldNumbers):
		scores=[]
		y=[]
		for i in range(0,FoldNumbers):
			if i != TestFold:
				scores+=( list(MatchScores_folds[i]) + list(MisMatchScores_folds[i]))
				y+=( [1]*len(MatchScores_folds[i]) + [0]*len(MisMatchScores_folds[i]) )
		fpr, tpr, thresholds = roc_curve(y, scores)				
		optimal_idx = np.argmax(tpr - fpr)
		optimal_threshold = thresholds[optimal_idx]
	
		scores=( list(MatchScores_folds[TestFold]) + list(MisMatchScores_folds[TestFold]))
		y=( [1]*len(MatchScores_folds[TestFold]) + [0]*len(MisMatchScores_folds[TestFold]) )
	
		TA=0
		TR=0
		FA=0
		FR=0
		for i in range(0,len(y)):
			if (scores[i] >= optimal_threshold and y[i]==1):
				TA+=1
			if (scores[i] >= optimal_threshold and y[i]==0):
				FA+=1
			if (scores[i] < optimal_threshold and y[i]==1):
				FR+=1
			if (scores[i] < optimal_threshold and y[i]==0):
				TR+=1
		

		
		Accuracy= (TA+TR)/(TA+TR+FA+FR)
		TA = TA/len(MatchScores_folds[TestFold])
		FA = FA/len(MisMatchScores_folds[TestFold])
		FR = FR/len(MatchScores_folds[TestFold])
		TR = TR/len(MisMatchScores_folds[TestFold])
		
		AvgTA+=TA
		AvgFA+=FA
		AvgTR+=TR
		AvgFR+=FR
		AvgAcc+=Accuracy
	

	Accuracy = (AvgAcc/FoldNumbers)
	TA = (AvgTA/FoldNumbers)
	TR = (AvgTR/FoldNumbers)
	FA = (AvgFA/FoldNumbers)
	FR = (AvgFR/FoldNumbers)
	
	print ("Accuracy: " + str(Accuracy))
	print ("TA Rate: " + str(TA))
	print ("TR Rate: " + str(TR))
	print ("FA Rate: " + str(FA))
	print ("FR Rate: " + str(FR))
	
	
def GetFullAcc():
	MatchScores = list(np.load("MatchScores.npy"))
	MisMatchScores = list(np.load("MisMatchScores.npy"))
	scores=( list(MatchScores) + list(MisMatchScores))
	y=( [1]*len(MatchScores) + [0]*len(MisMatchScores) )
	
	fpr, tpr, thresholds = roc_curve(y, scores)		
	
	plt.figure()
	lw = 2
	plt.plot(
	    fpr,
	    tpr,
	    color="darkorange",
	    lw=lw,
	)
	plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Receiver operating characteristic example")
	plt.legend(loc="lower right")
	plt.show()	
	
	
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	FPR = fpr[optimal_idx]
	TPR = tpr[optimal_idx]
	FNR = 1-TPR
	TNR = 1-FPR
	

	TP = TPR*len(MatchScores)
	TN = TNR*len(MisMatchScores)
	FP = FPR*len(MisMatchScores)
	FN = FNR*len(MatchScores)
	Accuracy = (TP+TN)/(FP+FN+TP+TN)
	
	print ("Accuracy: " + str(Accuracy))
	print ("TA Rate: " + str(TPR))
	print ("TR Rate: " + str(TNR))
	print ("FA Rate: " + str(FPR))
	print ("FR Rate: " + str(FNR))
	
GetScores()
#GetAccuracyKFolds(2)
#GetFullAcc()
