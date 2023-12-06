import numpy as np
from tensorflow import keras
import matplotlib.pyplot as pp
from tensorflow.python.ops.numpy_ops import np_config
from NeuralNetworkCoordinates import NetworkCoordinates, LayerCoordinates, NeuronCoordinates

class NNVisualiser:

  '''-----------------------------Class variables----------------------------'''
  __beautifyPlotDictionary = dict();
  __beautifyPlotDictionary['legend'] = __beautifyPlotDictionary['grid'] = __beautifyPlotDictionary['xyAxes'] = True;
  __beautifyPlotDictionary['xLowerLimit'] = __beautifyPlotDictionary['yLowerLimit'] = -1;
  __beautifyPlotDictionary['xUpperLimit'] = __beautifyPlotDictionary['yUpperLimit'] = 1;
  __savePlots:bool = False;
  __filePath:str = "./";
  __fileFormat:str = 'png';
  __gridPlot:bool = False;
  '''--------------------------Class Methods - Public------------------------'''
  @classmethod
  def getBeautifyPlotParameters(cls):
    return list(cls.__beautifyPlotDictionary.items());
  @classmethod
  def getBeautifyPlotParameter(cls, parameter:str):
    return cls.__beautifyPlotDictionary[parameter];
  @classmethod
  def setBeautifyPlotParameters(cls, **kwargs):
    for key, value in kwargs.items():
      if key in ['legend', 'grid', 'xyAxes'] and type(value) == bool:
        cls.__beautifyPlotDictionary[key] = value;
      elif key in ['xLowerLimit', 'xUpperLimit', 'yLowerLimit', 'yUpperLimit'] and type(value) == int:
        cls.__beautifyPlotDictionary[key] = value;
      else:
        return False;
      return True;
  @classmethod
  def isSavePlots(cls):
    return cls.__savePlots;
  @classmethod
  def setSavePlots(cls, savePlots:bool=False):
    cls.__savePlots = savePlots;
    return True;
  @classmethod
  def getFilePath(cls):
    return cls.__filePath;
  @classmethod
  def setFilePath(cls, filePath:str="./"):
    if filePath is not None:
      cls.__filePath = filePath;
      return True;
    return False;
  @classmethod
  def getFileFormat(cls):
    return cls.__fileFormat;
  @classmethod
  def setFileFormat(cls, fileFormat:str = 'png'):
    if fileFormat in ['png', 'pdf']:
      cls.__fileFormat = fileFormat;
      return True;
    return False;
  @classmethod
  def __isGridPlot(cls):
    return cls.__gridPlot;
  @classmethod
  def __setGridPlot(cls, gridPlot:bool=False):
    cls.__gridPlot = gridPlot;
    return True;

  def __init__(self, model:keras.Model, initialInput:np.ndarray=[], trainingData:np.ndarray=None):
    np_config.enable_numpy_behavior()
    self.__networkCoordinates = NetworkCoordinates(model, initialInput);
    self.__trainingData = trainingData;
    self.__networkCoordinates.prepareCoordinates();


  '''-------------------------Instance Private Methods------------------------'''
  def __isValidLayer(self, layer:int):
    return layer >= 0 and layer < len(self.__networkCoordinates.getLayerCoordinatesList());

  def __isValidNeuron(self, layer:int, neuron:int):
    if self.__isValidLayer(layer):
      return neuron >= 0 and neuron < len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList());

  '''-------------------------Instance Public Methods------------------------'''
  def getNetworkCoordinates(self):
    return self.__networkCoordinates;

  def getInitialInput(self):
    return self.__networkCoordinates.getInitialInput();

  def setInitialInput(self, initialInput:np.array):
    return self.__networkCoordinates.setInitialInput(initialInput);

  def getTrainingData(self):
    return self.__trainingData;

  def setTrainingData(self, trainingData:np.ndarray):
    if trainingData is not None and np.array(trainingData).size != 0:
      self.__trainingData = np.array(trainingData);
      return True;
    return False;

  '''Get Methods of Plot Coordinates'''
  def getLayerInputPlotCoordinates(self, layer:int):
    coordinatesList = [];
    if self.__isValidLayer(layer):
      coordinatesList.append((self.__networkCoordinates.getInitialInput().T, self.__networkCoordinates.getLayerCoordinates(layer).getInputMatrix().T));
    return coordinatesList;

  def getScalePlotCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
      coordinatesList.append((self.__networkCoordinates.getInitialInput().T, np.reshape(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getScaledVector().T, (np.size(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getScaledVector()), 1))));
    return coordinatesList;

  def getIndividualScalePlotCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
        coordinatesList.append((self.__networkCoordinates.getInitialInput().T, self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getIndividualScaleMatrix().T));
    return coordinatesList;

  def getLayerScalePlotIndividualCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
      coordinatesList = [None]*len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList());
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        coordinatesList[neuron]=self.getIndividualScalePlotCoordinates(layer, neuron);
    return coordinatesList;

  def getLayerScalePlotCoordinates(self, layer:int):
    coordinatesList = [];
    if self.__isValidLayer(layer):
      coordinatesList = [None]*len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList());
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        coordinatesList[neuron] = self.getScalePlotCoordinates(layer, neuron);
    return coordinatesList;

  def getLayerTranslationPlotCoordinates(self, layer:int):
    coordinatesList = [];
    if self.__isValidLayer(layer):
      coordinatesList = [None]*len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList());
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        coordinatesList[neuron] = self.getTranslationPlotCoordinates(layer, neuron);
    return coordinatesList;

  def getLayerActivationPlotCoordinates(self, layer:int):
    coordinatesList = [];
    if self.__isValidLayer(layer):
      coordinatesList = [None]*len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList());
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        coordinatesList[neuron] = self.getActivationPlotCoordinates(layer, neuron);
    return coordinatesList;

  def getCumulativeScalePlotCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
      coordinatesList.append((self.__networkCoordinates.getInitialInput().T, self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getCumulativeScaleMatrix().T));
    return coordinatesList;

  def getTranslationPlotCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
      coordinatesList.append((self.__networkCoordinates.getInitialInput().T, np.reshape(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getTranslatedVector().T, (np.size(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getTranslatedVector()),1))));
    return coordinatesList;

  def getActivationPlotCoordinates(self, layer:int, neuron:int):
    coordinatesList = [];
    if self.__isValidNeuron(layer, neuron):
      coordinatesList.append((self.__networkCoordinates.getInitialInput().T, np.reshape(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getActivatedVector().T, (np.size(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinates(neuron).getActivatedVector()),1))));
    return coordinatesList;


  '''Plot Methods'''
  def plotInputIndividualForNeuron(self, layer:int, neuron:int=0):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Input Plot - Individual - Layer "+str(layer)+"- Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getLayerInputPlotCoordinates(layer), plotTitle);

  def plotInputConsolidatedForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Input Plot - Consolidated - Layer "+str(layer)+"- Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getLayerInputPlotCoordinates(layer), plotTitle);

  def plotInputIndividualForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      plotTitle = "Input Plot - Individual - Layer "+str(layer);
      NNVisualiser.__plotIndividualForNeuron(self.getLayerInputPlotCoordinates(layer), plotTitle);

  def plotInputConsolidatedForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      plotTitle = "Input Plot - Consolidated - Layer "+str(layer);
      NNVisualiser.__plotConsolidatedForNeuron(self.getLayerInputPlotCoordinates(layer), plotTitle);

  def plotInputIndividualForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotInputIndividualForLayer(layer);

  def plotInputConsolidatedForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotInputConsolidatedForLayer(layer);

  def plotScaleForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Scale Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getScalePlotCoordinates(layer, neuron), plotTitle, "Scaled");

  def plotScaleForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotScaleForNeuron(layer, neuron);

  def plotScaleForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotScaleForLayer(layer);

  def plotScaleConsolidatedForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      plotTitle = "Scale Plot - Consolidated - Layer "+str(layer);
      NNVisualiser.__plotConsolidatedForLayer(self.getLayerScalePlotCoordinates(layer), plotTitle);

  def plotScaleConsolidatedForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotScaleConsolidatedForLayer(layer);

  def plotIndividualScaleForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Scale Plot - Individual - Layer "+str(layer)+" - Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getIndividualScalePlotCoordinates(layer, neuron), plotTitle, "Scaled");

  def plotIndividualScaleForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotIndividualScaleForNeuron(layer, neuron);

  def plotIndividualScaleForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotIndividualScaleForLayer(layer);

  def plotConsolidatedScaleForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Individual Scale Plot - Consolidated - Layer "+str(layer)+" - Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getIndividualScalePlotCoordinates(layer, neuron), plotTitle);

  def plotConsolidatedScaleForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotConsolidatedScaleForNeuron(layer, neuron);

  def plotConsolidatedScaleForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotConsolidatedScaleForLayer(layer);

  def plotIndividualCumulativeScaleForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Cumulative Scale Plot - Individual - Layer "+str(layer)+" - Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getCumulativeScalePlotCoordinates(layer, neuron), plotTitle);

  def plotIndividualCumulativeScaleForLayer(self, layer:int):
    for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
      self.plotIndividualCumulativeScaleForNeuron(layer, neuron);

  def plotIndividualCumulativeScaleForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotIndividualCumulativeScaleForLayer(layer);

  def plotConsolidatedCumulativeScaleForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Cumulative Scale Plot - Consolidated - Layer "+str(layer)+" - Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getCumulativeScalePlotCoordinates(layer, neuron), plotTitle);

  def plotConsolidatedCumulativeScaleForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotConsolidatedCumulativeScaleForNeuron(layer, neuron);

  def plotConsolidatedCumulativeScaleForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotConsolidatedCumulativeScaleForLayer(layer);

  def plotTranslationForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Scale Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getTranslationPlotCoordinates(layer, neuron), plotTitle, "Translated");

  def plotTranslationForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotTranslationForNeuron(layer, neuron);

  def plotTranslationForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotTranslationForLayer(layer);

  def plotTranslationConsolidatedForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      plotTitle = "Translation Plot - Consolidated - Layer "+str(layer);
      NNVisualiser.__plotConsolidatedForLayer(self.getLayerTranslationPlotCoordinates(layer), plotTitle);

  def plotTranslationConsolidatedForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotTranslationConsolidatedForLayer(layer);

  def plotActivationForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      plotTitle = "Scale Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotIndividualForNeuron(self.getActivationPlotCoordinates(layer, neuron), plotTitle, "Activated");

  def plotActivationForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotActivationForNeuron(layer, neuron);

  def plotActivationForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotActivationForLayer(layer);

  def plotActivationConsolidatedForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      plotTitle = "Activation Plot - Consolidated - Layer "+str(layer);
      NNVisualiser.__plotConsolidatedForLayer(self.getLayerActivationPlotCoordinates(layer), plotTitle);

  def plotActivationConsolidatedForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotActivationConsolidatedForLayer(layer);

  def plotFlowForNeuron(self, layer:int, neuron:int):
    if self.__isValidNeuron(layer, neuron):
      figure, axes = NNVisualiser.__plotGrid(row = 1, column = 5);
      superTitle = 'Flow plot of Layer '+str(layer)+' Neuron '+str(neuron);
      figure.suptitle(superTitle);

      # Consolidated Input Plot
      plotTitle = "Input Plot - Consolidated - Layer "+str(layer)+"- Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getLayerInputPlotCoordinates(layer), plotTitle, axes = axes[0]);

      # Individual Scale Consolidated Plot
      plotTitle = "Individual Scale Plot - Consolidated - Layer "+str(layer)+" - Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getIndividualScalePlotCoordinates(layer, neuron), plotTitle, axes = axes[1]);

      # Scale Plot
      plotTitle = "Scale Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getScalePlotCoordinates(layer, neuron), plotTitle, "Scaled", axes[2]);

      # Translation Plot
      plotTitle = "Translation Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getTranslationPlotCoordinates(layer, neuron), plotTitle, "Translated", axes[3]);

      # Activation Plot
      plotTitle = "Activation Plot - Layer "+str(layer)+" Neuron "+str(neuron);
      NNVisualiser.__plotConsolidatedForNeuron(self.getActivationPlotCoordinates(layer, neuron), plotTitle, "Activated", axes[4]);

      NNVisualiser.__savePlot(superTitle);
      # Set gridPlot to False
      NNVisualiser.__setGridPlot(False);

  def plotFlowForLayer(self, layer:int):
    if self.__isValidLayer(layer):
      for neuron in range(len(self.__networkCoordinates.getLayerCoordinates(layer).getNeuronCoordinatesList())):
        self.plotFlowForNeuron(layer, neuron);

  def plotFlowForNetwork(self):
    for layer in range(len(self.__networkCoordinates.getLayerCoordinatesList())):
      self.plotFlowForLayer(layer);

  '''Plot Methods - Static'''
  @staticmethod
  def __plotIndividual(X:np.ndarray, y:np.ndarray, label:str=""):
    figure, axes = pp.subplots();
    return NNVisualiser.__plotConsolidated(X, y, label, axes);

  @staticmethod
  def __plotConsolidated(X:np.ndarray, y:np.ndarray, label:str="", axes:pp.Axes=None):
    if axes is None:
      axes = NNVisualiser.__plotIndividual(X, y, label);
    else:
      axes.plot(X, y, label=label);
    return axes;

  @staticmethod
  def __beautifyPlot(axes:pp.Axes, title:str="Plot"):
    axes.set_xlim(NNVisualiser.getBeautifyPlotParameter('xLowerLimit'), NNVisualiser.getBeautifyPlotParameter('xUpperLimit'));
    axes.set_ylim(NNVisualiser.getBeautifyPlotParameter('yLowerLimit'), NNVisualiser.getBeautifyPlotParameter('yUpperLimit'));
    axes.set_title(title);
    if NNVisualiser.getBeautifyPlotParameter('grid'):
      axes.grid(True);
    if NNVisualiser.getBeautifyPlotParameter('legend'):
      axes.legend();
    if NNVisualiser.getBeautifyPlotParameter('xyAxes'):
      axes.axhline(0, color='black');
      axes.axvline(0, color='black');
    if not NNVisualiser.__isGridPlot():
      NNVisualiser.__savePlot(title);

  @staticmethod
  def __savePlot(title:str="Plot"):
    if NNVisualiser.isSavePlots():
      NNVisualiser.__fileSave(title);

  @staticmethod
  def __plotIndividualForNeuron(plotCoordinates, title="Individual Neuron Plot ", label:str="Input 0"):
    for X, y in plotCoordinates:
      for input in range(np.shape(y)[1]):
        if np.shape(y)[1] != 1:
          label = "Input "+str(input);
        NNVisualiser.__beautifyPlot(NNVisualiser.__plotIndividual(X, y[:, input], label), title +" "+ label);

  @staticmethod
  def __plotConsolidatedForNeuron(plotCoordinates, title="Consolidated Neuron Plot ", label:str="", axes:pp.Axes=None):
    for X, y in plotCoordinates:
      for input in range(np.shape(y)[1]):
        if label == "":
          axes = NNVisualiser.__plotConsolidated(X, y[:, input], "Input "+str(input), axes);
        else:
          axes = NNVisualiser.__plotConsolidated(X, y[:, input], label, axes);
    NNVisualiser.__beautifyPlot(axes, title);

  @staticmethod
  def __plotIndividualForLayer(plotCoordinatesList, title="Individual Plot - Layer "):
    for neuron in range(len(plotCoordinatesList)):
      plotCoordinates = plotCoordinatesList[neuron];
      for X, y in plotCoordinates:
        label = "Neuron "+str(neuron);
        NNVisualiser.__beautifyPlot(NNVisualiser.__plotIndividual(X, y, label), title +" "+ label);

  @staticmethod
  def __plotConsolidatedForLayer(plotCoordinatesList, title="Consolidated Plot - Layer", axes:pp.Axes=None):
    for neuron in range(len(plotCoordinatesList)):
      plotCoordinates = plotCoordinatesList[neuron];
      for X, y in plotCoordinates:
        label = "Neuron "+str(neuron);
        axes = NNVisualiser.__plotConsolidated(X, y, label, axes);
      NNVisualiser.__beautifyPlot(axes, title);

  @staticmethod
  def __plotGrid(row:int, column:int, sharex=True, sharey=True, figsize=(30, 5)):
    figure, axes = pp.subplots(row, column, sharex=sharex, sharey=sharey, figsize=figsize);
    NNVisualiser.__setGridPlot(True);
    return figure, axes;

  @staticmethod
  def __fileSave(title:str):
    pp.savefig(str(NNVisualiser.getFilePath())+title+"."+NNVisualiser.getFileFormat(), format=NNVisualiser.getFileFormat());
