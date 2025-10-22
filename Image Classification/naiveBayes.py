import util
import math
import statistics

class NaiveBayesClassifier:

  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"

    self.covertedFaceProbability = 0.0
    self.covertedNotFacerobability = 0.0

    self.priorTrue = 0
    self.priorFalse = 0
    
    self.covertedZeroProbability = 0.0001
    self.covertedOneProbability = 0.0001
    self.covertedTwoProbability = 0.0001
    self.covertedThreeProbability = 0.0001
    self.covertedFourProbability = 0.0001
    self.covertedFiveProbability = 0.0001
    self.covertedSixProbability = 0.0001
    self.covertedSevenProbability = 0.0001
    self.covertedEightProbability = 0.0001
    self.covertedNineProbability = 0.0001

    self.priorNumber = [0 for _ in range(10)]


  def train(self, trainingData, trainingLabels, validationData, validationLabels, featureSize, datumSizeX, datumSizeY):
  
    section_size = featureSize

    num_sections_x = datumSizeX // section_size
    num_sections_y = datumSizeY // section_size

    featureCount = num_sections_x * num_sections_y
    maxFeatureSize = featureSize * featureSize * 2
  
    if datumSizeX == 60:
      FaceFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      NotFaceFeatureCounter = [[0] * 50 for _ in range(featureCount)]

      self.ConvertedFaceProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedNotFaceProbability = [[0] * 50 for _ in range(featureCount)]

      totalTrue = 0
      for label in trainingLabels:
        if label == 1:
          totalTrue += 1
        
      priorTrue = totalTrue / len(trainingLabels)
      priorFalse = 1 - priorTrue
      
      acc = []
      
      for i in range(len(trainingData)):
        featureNumber = 0
        for section_x in range(num_sections_x):
          for section_y in range(num_sections_y):
            start_x = section_x * section_size
            end_x = start_x + section_size
            start_y = section_y * section_size
            end_y = start_y + section_size
   
            featureValue = sum(trainingData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
            
            if trainingLabels[i] == 1:
              FaceFeatureCounter[featureNumber][featureValue] += 1
            else:
              NotFaceFeatureCounter[featureNumber][featureValue] += 1
                
            featureNumber += 1

      for i in range(len(FaceFeatureCounter)):
        for j in range(len(FaceFeatureCounter[i])):
          if FaceFeatureCounter[i][j] != 0:
            self.ConvertedFaceProbability[i][j] = FaceFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedFaceProbability[i][j] = 0.001
            
          if NotFaceFeatureCounter[i][j] != 0:
            self.ConvertedNotFaceProbability[i][j] = NotFaceFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedFaceProbability[i][j] = 0.001

      for i in range(len(validationData)):
        probabilityXGivenTrue = 1.0
        probabilityXGivenFalse = 1.0
        featureNumber = 0
        
        for section_x in range(num_sections_x):
          for section_y in range(num_sections_y):
            start_x = section_x * section_size
            end_x = start_x + section_size
            start_y = section_y * section_size
            end_y = start_y + section_size

            featureValue = sum(validationData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))

            probabilityXGivenTrue *= self.ConvertedFaceProbability[featureNumber][featureValue]
            probabilityXGivenFalse *= self.ConvertedNotFaceProbability[featureNumber][featureValue]
            
            featureNumber += 1

        
        probabilityImageTrue = probabilityXGivenTrue * priorTrue
        probabilityImageFalse = probabilityXGivenFalse * priorFalse

        predicted_label = 1 if probabilityImageTrue > probabilityImageFalse else 0

        if (predicted_label == validationLabels[i]):
          acc.append(1)
        else:
          acc.append(0)

      
      print("Mean: ", statistics.mean(acc) * 100)
      print("Standard Deviation: ", statistics.stdev(acc) * 100)
    else:
      zeroFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      oneFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      twoFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      threeFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      fourFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      fiveFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      sixFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      sevenFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      eightFeatureCounter = [[0] * 50 for _ in range(featureCount)]
      nineFeatureCounter = [[0] * 50 for _ in range(featureCount)]
    
      self.ConvertedZeroProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedOneProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedTwoProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedThreeProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedFourProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedFiveProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedSixProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedSevenProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedEightProbability = [[0] * 50 for _ in range(featureCount)]
      self.ConvertedNineProbability = [[0] * 50 for _ in range(featureCount)]

      totalLabels = [0 for _ in range(10)]
      
      for label in trainingLabels:
        totalLabels[label] += 1

      priorNumber = [0 for _ in range(10)]

      for i in range(10):
        self.priorNumber[i] = totalLabels[i] / len(trainingLabels)
      
      acc = []
      
      for i in range(len(trainingData)):
        featureNumber = 0
        for section_x in range(num_sections_x):
          for section_y in range(num_sections_y):
            start_x = section_x * section_size
            end_x = start_x + section_size
            start_y = section_y * section_size
            end_y = start_y + section_size
   
            featureValue = sum(trainingData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
            
            if trainingLabels[i] == 0:
              zeroFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 1:
              oneFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 2:
              twoFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 3:
              threeFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 4:
              fourFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 5:
              fiveFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 6:
              sixFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 7:
              sevenFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 8:
              eightFeatureCounter[featureNumber][featureValue] += 1
            if trainingLabels[i] == 9:
              nineFeatureCounter[featureNumber][featureValue] += 1
            featureNumber += 1
      for i in range(len(zeroFeatureCounter)):
        for j in range(len(zeroFeatureCounter[i])):
          if zeroFeatureCounter[i][j] != 0:
            self.ConvertedZeroProbability[i][j] = zeroFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedZeroProbability[i][j] = 0.001
            
          if oneFeatureCounter[i][j] != 0:
            self.ConvertedOneProbability[i][j] = oneFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedOneProbability[i][j] = 0.001
            
          if twoFeatureCounter[i][j] != 0:
            self.ConvertedTwoProbability[i][j] = twoFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedTwoProbability[i][j] = 0.001
          
          if threeFeatureCounter[i][j] != 0:
            self.ConvertedThreeProbability[i][j] = threeFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedThreeProbability[i][j] = 0.001
            
          if fourFeatureCounter[i][j] != 0:
            self.ConvertedFourProbability[i][j] = fourFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedFourProbability[i][j] = 0.001
            
          if fiveFeatureCounter[i][j] != 0:
            self.ConvertedFiveProbability[i][j] = fiveFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedFiveProbability[i][j] = 0.001
            
          if sixFeatureCounter[i][j] != 0:
            self.ConvertedSixProbability[i][j] = sixFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedSixProbability[i][j] = 0.001
            
          if sevenFeatureCounter[i][j] != 0:
            self.ConvertedSevenProbability[i][j] = sevenFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedSevenProbability[i][j] = 0.001
            
          if eightFeatureCounter[i][j] != 0:
            self.ConvertedEightProbability[i][j] = eightFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedEightProbability[i][j] = 0.001
            
          if nineFeatureCounter[i][j] != 0:
            self.ConvertedNineProbability[i][j] = nineFeatureCounter[i][j] / maxFeatureSize
          else:
            self.ConvertedNineProbability[i][j] = 0.001
          
      for i in range(len(validationData)):
        probabilityXGivenNumber = [1.0 for x in range(10)]
        featureNumber = 0
        
        for section_x in range(num_sections_x):
          for section_y in range(num_sections_y):
            start_x = section_x * section_size
            end_x = start_x + section_size
            start_y = section_y * section_size
            end_y = start_y + section_size

            featureValue = sum(validationData[i].getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))

            for number in range(len(probabilityXGivenNumber)):

              if number == 0:
                probabilityXGivenNumber[number] *= self.ConvertedZeroProbability[featureNumber][featureValue]
                
              if number == 1:
                probabilityXGivenNumber[number] *= self.ConvertedOneProbability[featureNumber][featureValue]
              if number == 2:
                probabilityXGivenNumber[number] *= self.ConvertedTwoProbability[featureNumber][featureValue]
              if number == 3:
                probabilityXGivenNumber[number] *= self.ConvertedThreeProbability[featureNumber][featureValue]
              if number == 4:
                probabilityXGivenNumber[number] *= self.ConvertedFourProbability[featureNumber][featureValue]
              if number == 5:
                probabilityXGivenNumber[number] *= self.ConvertedFiveProbability[featureNumber][featureValue]
              if number == 6:
                probabilityXGivenNumber[number] *= self.ConvertedSixProbability[featureNumber][featureValue]
              if number == 7:
                probabilityXGivenNumber[number] *= self.ConvertedSevenProbability[featureNumber][featureValue]
              if number == 8:
                probabilityXGivenNumber[number] *= self.ConvertedEightProbability[featureNumber][featureValue]
              if number == 9:
                probabilityXGivenNumber[number] *= self.ConvertedNineProbability[featureNumber][featureValue]
            featureNumber += 1

        probabilityImageNumber = [1.0 for x in range(10)]

        for number in range(len(probabilityImageNumber)):

          probabilityImageNumber[number] = probabilityXGivenNumber[number] * self.priorNumber[number]
          predicted_label = probabilityImageNumber.index(max(probabilityImageNumber))

        if (predicted_label == validationLabels[i]):
          acc.append(1)
        else:
          acc.append(0)
       
      print("Mean: ", statistics.mean(acc) * 100)
      print("Standard Deviation: ", statistics.stdev(acc) * 100)
      
  def classify(self, datum, datum_label, featureSize, datumSizeX, datumSizeY):
    section_size = featureSize
    featureNumber = 0
    num_sections_x = datumSizeX // section_size
    num_sections_y = datumSizeY // section_size
    
    if datumSizeX == 60:
      probabilityXGivenTrue = 1.0
      probabilityXGivenFalse = 1.0
      
      for section_x in range(num_sections_x):
        for section_y in range(num_sections_y):
          start_x = section_x * section_size
          end_x = start_x + section_size
          start_y = section_y * section_size
          end_y = start_y + section_size

          featureValue = sum(datum.getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))
          probabilityXGivenTrue *= self.ConvertedFaceProbability[featureNumber][featureValue]
          probabilityXGivenFalse *= self.ConvertedNotFaceProbability[featureNumber][featureValue]
            
          featureNumber += 1

        
      probabilityImageTrue = probabilityXGivenTrue * self.priorTrue
      probabilityImageFalse = probabilityXGivenFalse * self.priorFalse

      predicted_label = 1 if probabilityImageTrue > probabilityImageFalse else 0
      if (predicted_label == 0):
        print("Predicted: This is not a face")
      else:
        print("Predicted: This is a face")
      if (datum_label == 0):
        print("Answer:    This is not a face")
      else:
        print("Answer:    This is a face")
          
    else:
      probabilityXGivenNumber = [1.0 for x in range(10)]
      
      for section_x in range(num_sections_x):
        for section_y in range(num_sections_y):
          start_x = section_x * section_size
          end_x = start_x + section_size
          start_y = section_y * section_size
          end_y = start_y + section_size

          featureValue = sum(datum.getPixel(x, y) for x in range(start_x, end_x) for y in range(start_y, end_y))

          for number in range(len(probabilityXGivenNumber)):

            if number == 0:
              probabilityXGivenNumber[number] *= self.ConvertedZeroProbability[featureNumber][featureValue]
            if number == 1:
              probabilityXGivenNumber[number] *= self.ConvertedOneProbability[featureNumber][featureValue]
            if number == 2:
              probabilityXGivenNumber[number] *= self.ConvertedTwoProbability[featureNumber][featureValue]
            if number == 3:
              probabilityXGivenNumber[number] *= self.ConvertedThreeProbability[featureNumber][featureValue]
            if number == 4:
              probabilityXGivenNumber[number] *= self.ConvertedFourProbability[featureNumber][featureValue]
            if number == 5:
              probabilityXGivenNumber[number] *= self.ConvertedFiveProbability[featureNumber][featureValue]
            if number == 6:
              probabilityXGivenNumber[number] *= self.ConvertedSixProbability[featureNumber][featureValue]
            if number == 7:
              probabilityXGivenNumber[number] *= self.ConvertedSevenProbability[featureNumber][featureValue]
            if number == 8:
              probabilityXGivenNumber[number] *= self.ConvertedEightProbability[featureNumber][featureValue]
            if number == 9:
              probabilityXGivenNumber[number] *= self.ConvertedNineProbability[featureNumber][featureValue]
          featureNumber += 1

      probabilityImageNumber = [1.0 for x in range(10)]

      for number in range(len(probabilityImageNumber)):

        probabilityImageNumber[number] = probabilityXGivenNumber[number] * self.priorNumber[number]
        predicted_label = probabilityImageNumber.index(max(probabilityImageNumber))

      print("Predicted: ", predicted_label)
      print("Answer:    ", datum_label)
