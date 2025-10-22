import util
import random
import statistics
class PerceptronClassifier:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter({(x, y): random.randint(-1000, 1000) for x in range(28) for y in range(28)})

        
    def train(self, trainingData, trainingLabels, validationData, validationLabels, featureSize, datumSizeX, datumSizeY):

        section_size = featureSize

        num_sections_x = datumSizeX // section_size
        num_sections_y = datumSizeY // section_size

        acc = []
        
        for iteration in range(self.max_iterations):
            for i in range(len(trainingData)):
                highest_score = float('-inf')
                predicted_label = None
                for label in self.legalLabels:
                    score = 0
                    for section_x in range(num_sections_x):
                        for section_y in range(num_sections_y):
                            start_x = section_x * section_size
                            end_x = start_x + section_size
                            start_y = section_y * section_size
                            end_y = start_y + section_size
                            
                            num_chars = sum(trainingData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
                            num_empty = section_size ** 2 - num_chars
                            
                            weight = self.weights[label][(section_x, section_y)]
                            score += (num_chars - num_empty) * weight
                            
                    if score > highest_score:
                        highest_score = score
                        predicted_label = label

                true_label = trainingLabels[i]

                if predicted_label != true_label:
                    for section_x in range(num_sections_x):
                        for section_y in range(num_sections_y):
                            start_x = section_x * section_size
                            end_x = start_x + section_size
                            start_y = section_y * section_size
                            end_y = start_y + section_size
                            
                            num_chars = sum(trainingData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
                            num_empty = section_size ** 2 - num_chars
                            
                            self.weights[true_label][(section_x, section_y)] += (num_chars - num_empty)
                            self.weights[predicted_label][(section_x, section_y)] -= (num_chars - num_empty)
            for i in range(len(validationData)):
                highest_score = float('-inf')
                predicted_label = None
                for label in self.legalLabels:
                    score = 0
                    for section_x in range(num_sections_x):
                        for section_y in range(num_sections_y):
                            start_x = section_x * section_size
                            end_x = start_x + section_size
                            start_y = section_y * section_size
                            end_y = start_y + section_size
                            
                            num_chars = sum(validationData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
                            num_empty = section_size ** 2 - num_chars
                            
                            weight = self.weights[label][(section_x, section_y)]
                            score += (num_chars - num_empty) * weight
                            
                    if score > highest_score:
                        highest_score = score
                        predicted_label = label
                if (predicted_label == validationLabels[i]):
                    acc.append(1)
                else:
                    acc.append(0)
                    
        print("Mean: ", statistics.mean(acc) * 100)
        print("Standard Deviation: ", statistics.stdev(acc) * 100)
            
    def classify(self, datum, datum_label, featureSize, datumSizeX, datumSizeY):
        section_size = featureSize

        num_sections_x = datumSizeX // section_size
        num_sections_y = datumSizeY // section_size

        highest_score = float('-inf')
        predicted_label = None
                              
        for l in self.legalLabels:
            score = 0
            for section_x in range(num_sections_x):
                for section_y in range(num_sections_y):
                    start_x = section_x * section_size
                    end_x = start_x + section_size
                    start_y = section_y * section_size
                    end_y = start_y + section_size
                                
                    num_chars = sum(datum.getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
                    num_empty = section_size ** 2 - num_chars
                            
                    weight = self.weights[l][(section_x, section_y)]
                    score += (num_chars - num_empty) * weight
                                
                if score > highest_score:
                    highest_score = score
                    predicted_label = l

        if (datumSizeX == 28):
            print("Predicted: ", predicted_label)
            print("Answer:    ", datum_label)
        else:
            if (predicted_label == 0):
                print("Predicted: This is not a face")
            else:
                print("Predicted: This is a face")
            if (datum_label == 0):
                print("Answer:    This is not a face")
            else:
                print("Answer:    This is a face")
