import util
import zipfile
import os
import random
from perceptron import PerceptronClassifier
from naiveBayes import NaiveBayesClassifier

DATUM_WIDTH = 0
DATUM_HEIGHT = 0 

class Datum:
    def __init__(self, data, width, height):
        global DATUM_WIDTH, DATUM_HEIGHT
        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data is None:
            data = [[' ' for _ in range(DATUM_WIDTH)] for _ in range(DATUM_HEIGHT)]
        self.pixels = util.arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):
        return self.pixels[column][row]

    def getAsciiString(self):
        rows = []
        data = util.arrayInvert(self.pixels)
        for row in data:
            ascii_chars = map(asciiGrayscaleConversionFunction, row)
            rows.append("".join(ascii_chars))
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()

    def rotate(self):
        self.pixels = [list(row) for row in zip(*self.pixels[::1])]

def loadDataFile(filename, n, width, height):
    global DATUM_WIDTH, DATUM_HEIGHT
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    
    for _ in range(n):
        data = []
        for _ in range(height):
            
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH - 1:

            print("Truncating at %d examples (maximum)" % len(items))
            break
        items.append(Datum(data, DATUM_WIDTH, DATUM_HEIGHT))
    return items

def readlines(filename):
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return [line.decode() for line in z.read(filename).split(b'\n')]

def loadLabelsFile(filename, n):
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

def asciiGrayscaleConversionFunction(value):
    if value == 0:
        return ' '
    elif value == 1:
        return '+'
    elif value == 2:
        return '#'
def IntegerConversionFunction(value):
    if value == ' ':
        return 0
    elif value == '+':
        return 1
    elif value == '#':
        return 2
    
def convertToInteger(data):
    if isinstance(data, str): 
        return IntegerConversionFunction(data)
    else:  
        return [convertToInteger(char) for char in data]

def testNumberPredictor(max_iterations, total, totalTraining):
    
    trainingItems = loadDataFile("data/digitdata/trainingimages", total, 28, 28)
    trainingLabels = loadLabelsFile("data/digitdata/traininglabels", total)

    randomTrainingItems = []
    randomTrainingLabels = []
    
    randomTraining = random.sample(range(5000), totalTraining)

    for index in randomTraining:
        randomTrainingItems.append(trainingItems[index])
        randomTrainingLabels.append(trainingLabels[index])
    

    validItems = loadDataFile("data/digitdata/validationimages", 1000, 28, 28)
    validLabels = loadLabelsFile("data/digitdata/validationlabels", 1000)
    
    NumberIndex = random.randint(0, 500)
    testItem = loadDataFile("data/digitdata/testimages", 500, 28, 28)[NumberIndex]
    trueLabel = loadLabelsFile("data/digitdata/testlabels", 500)[NumberIndex]

    print(testItem)
    
    print("")
    print("________________________________")
    print("perceptron")
    p = PerceptronClassifier(legalLabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], max_iterations = max_iterations)
    p.train(randomTrainingItems, randomTrainingLabels, validItems, validLabels, 4, 28, 28)
    predictedLabel = p.classify(testItem, trueLabel, 4, 28, 28)
    print("________________________________")
    
    print("")
    print("________________________________")
    print("naivebayes")
    n = NaiveBayesClassifier(legalLabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n.train(randomTrainingItems, randomTrainingLabels, validItems, validLabels, 4, 28, 28)
    predictedLabel = n.classify(testItem, trueLabel, 4, 28, 28)
    print("________________________________")
    print("")
    print("")
    


def testFaceDetector(max_iterations, total, totalTraining):
    
    trainingItems = loadDataFile("data/facedata/facedatatrain", total, 60, 70)
    trainingLabels = loadLabelsFile("data/facedata/facedatatrainlabels", total)

    randomTrainingItems = []
    randomTrainingLabels = []
    
    randomTraining = random.sample(range(400), totalTraining)

    for index in randomTraining:
        randomTrainingItems.append(trainingItems[index])
        randomTrainingLabels.append(trainingLabels[index])

    validItems = loadDataFile("data/facedata/facedatavalidation", 300, 60, 70)
    validLabels = loadLabelsFile("data/facedata/facedatavalidationlabels", 300)

    FaceIndex = random.randint(0, 150)
    testItem = loadDataFile("data/facedata/facedatatest", 150, 60, 70)[FaceIndex]
    trueLabel = loadLabelsFile("data/facedata/facedatatestlabels", 150)[FaceIndex]
    
    print(testItem)
    
    print("")
    print("________________________________")
    print("perceptron")
    p = PerceptronClassifier(legalLabels=[0, 1], max_iterations = max_iterations)
    p.train(randomTrainingItems, randomTrainingLabels, validItems, validLabels, 5, 60, 70)
    predictedLabel = p.classify(testItem, trueLabel, 5, 60, 70)
    print("________________________________")
    
    print("")

    print("________________________________")
    print("naivebayes")
    n = NaiveBayesClassifier(legalLabels=[0, 1])
    n.train(randomTrainingItems, randomTrainingLabels, validItems, validLabels, 5, 60, 70)
    predictedLabels = n.classify(testItem, trueLabel, 5, 60, 70)
    print("________________________________")
    print("")
    print("")
    


if __name__ == "__main__":
    
    percentage = 1
    
    totalTrainingNumber = int(5000.0 * percentage)
    testNumberPredictor(5, 5000, totalTrainingNumber)

    
    #totalTrainingFace = int(400.0 * percentage)
    #testFaceDetector(5, 400, totalTrainingFace)

    """
    for i in range(10):
        percentage = 0.1 * (i + 1)
        print("percentage: ", round(percentage * 100), "%")

        #totalTrainingNumber = int(5000.0 * percentage)
        #testNumberPredictor(5, 5000, totalTrainingNumber)

        totalTrainingFace = int(400.0 * percentage)
        testFaceDetector(5, 400, totalTrainingFace)
    """
