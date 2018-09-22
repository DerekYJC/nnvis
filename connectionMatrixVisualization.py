# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:59:10 2018

@author: derek
"""

import matplotlib.pyplot  as plt
import matplotlib         as mpl
import numpy              as np
from numpy.ma             import masked_array

# -----------------------------------------------------------------------------
'''
Create a Neural Network model with a dictionary.
  Example:
    {layer_i: # of neurons in layer_i} ...
'''
networkStructure    = {}
networkStructure[0] = 1   # One  neuron  in the 1st layer (Layer 0)
networkStructure[1] = 5   # Five neurons in the 2nd layer (Layer 1)
networkStructure[2] = 9   # Nine neurons in the 3rd layer (Layer 2)
networkStructure[3] = 2   # Two  neurons in the 4th layer (Layer 3)

# -----------------------------------------------------------------------------
'''
Obtain the dictionary that given # of layer and # of node, the dictionary would 
return the id of the neuron.
  Example:
    {(layer_i, neuron_j): id of the neuron} ... 
'''
_id = 0
neuronDict = {}
for layer in range(len(networkStructure)):
  for node in range(networkStructure[layer]):
    neuronDict[(layer, node)] = _id
    _id += 1
nNeurons = _id 
neuronLayer = list(networkStructure.values())

# -----------------------------------------------------------------------------
'''
Define the connection function to connect the neurons ...
  Example of connectionMatrix:
    connectionMatrix[from_neuron][to_neuron] = weight 
'''
connectionMatrix = np.zeros([nNeurons, nNeurons])
adaptableMatrix  = np.ones([nNeurons, nNeurons])
def connect(fromObject, toObject, typeofConnection='N2N', adaptivity=True, weight=1): 
  '''
  Purpose: Update connectionMatrix
  args:
    typeofConnection: a string. Ex: 'Neuron_to_Neuron', 'N2N', 
                                    'Neuron_to_Layer',  'N2L', 
                                    'Layer_to_Neuron', 'L2N', 
                                    'Layer_to_Layer', 'L2L'
    fromObject, toObject: connect from sth to sth.
    adaptivity: a boolean. 
    weight: a numerical value.
  '''
  # Transfer to neuron's ID
  if typeofConnection == 'Neuron_to_Neuron' or typeofConnection == 'N2N':
    if type(fromObject) == tuple:
      fromObject = neuronDict[fromObject]
    elif type(fromObject) == int:
      fromObject = fromObject
    elif type(fromObject) == list:
      if type(fromObject[0]) == tuple:
        fromobject = []
        for elem in fromObject:
          fromobject.append(neuronDict[elem])
        fromObject = fromobject
    else:
      raise('Connection is not recognizable')    
    # -------------------------------------------------------------------------
    if type(toObject) == tuple:
      toObject = neuronDict[toObject]
    elif type(toObject) == int:
      toObject = toObject
    elif type(toObject) == list:
      if type(toObject[0]) == tuple:
        toobject = []
        for elem in toObject:
          toobject.append(neuronDict[elem])
        toObject = toobject 
    else:
      raise('Connection is not recognizable')    
  
  elif typeofConnection == 'Neuron_to_Layer' or typeofConnection == 'N2L':
    if type(fromObject) == tuple:
      fromObject = neuronDict[fromObject]
    elif type(fromObject) == int:
      fromObject = fromObject
    elif type(fromObject) == list:
      if type(fromObject[0]) == tuple:
        fromobject = []
        for elem in fromObject:
          fromobject.append(neuronDict[elem])
        fromObject = fromobject
    else:
      raise('Connection is not recognizable')    
    # -------------------------------------------------------------------------
    if type(toObject) == int:
      toobject = []
      for elem in range(networkStructure[toObject]):
        toobject.append(neuronDict[(toObject, elem)])
      toObject = toobject
    elif type(toObject) == list:
      if type(toObject[0]) == int:
        toobject = []
        for layer in toObject:
          for elem in range(networkStructure[layer]):
            toobject.append(neuronDict[(layer, elem)]) 
        toObject = toobject
      else:
        raise('Connection is not recognizable')      
    else:
      raise('Connection is not recognizable')      
    
  elif typeofConnection == 'Layer_to_Neuron' or typeofConnection == 'L2N':
    if type(fromObject) == int:
      fromobject = []
      for elem in range(networkStructure[fromObject]):
        fromobject.append(neuronDict[(fromObject, elem)])
      fromObject = fromobject
    elif type(fromObject) == list:
      if type(fromObject[0]) == int:
        fromobject = []
        for layer in fromObject:
          for elem in range(networkStructure[layer]):
            fromobject.append(neuronDict[(layer, elem)]) 
        fromObject = fromobject
      else:
        raise('Connection is not recognizable')      
    else:
      raise('Connection is not recognizable')    
    # -------------------------------------------------------------------------
    if type(toObject) == tuple:
      toObject = neuronDict[toObject]
    elif type(toObject) == int:
      toObject = toObject
    elif type(toObject) == list:
      if type(toObject[0]) == tuple:
        toobject = []
        for elem in toObject:
          toobject.append(neuronDict[elem])
        toObject = toobject 
    else:
      raise('Connection is not recognizable')  
  
  elif typeofConnection == 'Layer_to_Layer' or typeofConnection == 'L2L':
    if type(fromObject) == int:
      fromobject = []
      for elem in range(networkStructure[fromObject]):
        fromobject.append(neuronDict[(fromObject, elem)])
      fromObject = fromobject
    elif type(fromObject) == list:
      if type(fromObject[0]) == int:
        fromobject = []
        for layer in fromObject:
          for elem in range(networkStructure[layer]):
            fromobject.append(neuronDict[(layer, elem)]) 
        fromObject = fromobject
      else:
        raise('Connection is not recognizable')      
    else:
      raise('Connection is not recognizable')   
    # -------------------------------------------------------------------------
    if type(toObject) == int:
      toobject = []
      for elem in range(networkStructure[toObject]):
        toobject.append(neuronDict[(toObject, elem)])
      toObject = toobject
    elif type(toObject) == list:
      if type(toObject[0]) == int:
        toobject = []
        for layer in toObject:
          for elem in range(networkStructure[layer]):
            toobject.append(neuronDict[(layer, elem)]) 
        toObject = toobject
      else:
        raise('Connection is not recognizable')      
    else:
      raise('Connection is not recognizable')   
      
  else:
    raise('Connection is not recognizable')   
  
  # Connect !!
  if type(fromObject) == int:
    connectionMatrix[fromObject][toObject] = weight
    adaptableMatrix[fromObject][toObject] = adaptivity
  else:
    for fromobject in fromObject:
      connectionMatrix[fromobject][toObject] = weight
      adaptableMatrix[fromobject][toObject] = adaptivity

# -----------------------------------------------------------------------------    
# Create some connection      
connect((2, 1), [(1, 3), (2, 4), (2, 5)], weight=-0.5)         # N2N
connect([(0, 0), (1, 0)], [1, 2], 'N2L', False, weight=0.75)   # N2L
connect(1, [(3, 0), (3, 1)], 'L2N', True, weight=1.25)         # L2N
connect(2, 3, 'L2L', False, weight=-1.5)                       # L2L
connect(1, [8, 9, 11, 12, 13], 'L2N', True, weight=-2)         # L2N
connect(2, [6, 7, 12], 'L2N', False, weight=3)                 # L2N
    
# -----------------------------------------------------------------------------
# Find the index that the connection has False adaptivity
def index_2d(myList, value):
  indexList = []
  for i, x in enumerate(myList):
    if value in x:
      indArray = np.where(x == value)[0]
      for elem in indArray:
        indexList.append((i, elem))
  return indexList
trueAdapt  = index_2d(adaptableMatrix, 1)
falseAdapt = index_2d(adaptableMatrix, 0)

# -----------------------------------------------------------------------------

def showConnection(connectionMatrix, adaptableMatrix, neuronLayer):
  
  nNeurons = np.shape(connectionMatrix)[0]
  
  # Use masked_array for two differen colormaps (adaptable or not)
  connectionMatrixF = masked_array(connectionMatrix, adaptableMatrix == 1)
  connectionMatrixT = masked_array(connectionMatrix, adaptableMatrix == 0)
  
  # Visualize the connection matrix with imshow ...
  plt.figure(figsize=(9,6))
  # For Not Adaptable
  cmapF = mpl.colors.LinearSegmentedColormap.from_list('', ['darkorange','white','red'])
  pF = plt.imshow(connectionMatrixF, cmap=cmapF)
  if adaptableMatrix.all():
    scaleF = 1
  else:
    scaleF = max(abs(connectionMatrixF.compressed()))
  cbF = plt.colorbar(pF)
  plt.clim(-scaleF, scaleF)
  cbF.set_label('Non-Adaptable', fontsize=15)
  # For Adaptable
  cmapT = mpl.colors.LinearSegmentedColormap.from_list('', ['dodgerblue','white','limegreen']) 
  pT = plt.imshow(connectionMatrixT, cmap=cmapT)
  if ~ adaptableMatrix.any():
    scaleT = 1
  else:
    scaleT = max(abs(connectionMatrixT.compressed()))
  cbT = plt.colorbar(pT)
  plt.clim(-scaleT, scaleT) 
  cbT.set_label('Adaptable', fontsize=15)
  # Set the tick labels 
  ax = plt.gca()
  ax.set_xticks(np.arange(0, nNeurons, 1))
  ax.set_yticks(np.arange(0, nNeurons, 1))
  ax.set_xticklabels(np.arange(nNeurons))
  ax.set_yticklabels(np.arange(nNeurons))
  ax.tick_params(labelbottom=False, labeltop=True)
  ax.tick_params(axis='both', which='both', length=0)
  # Minor ticks (Separate each neuron with each other)
  ax.set_xticks(np.arange(-.5, nNeurons, 1), minor=True);
  ax.set_yticks(np.arange(-.5, nNeurons, 1), minor=True);
  # Gridlines based on minor ticks
  ax.grid(which='minor', color='w', linestyle='-', linewidth=nNeurons/10)
  ax.set_aspect('equal')
  # Separate by each Layer
  for layer in range(len(neuronLayer)):
    plt.plot([-.5+np.sum(neuronLayer[:layer]), -.5+np.sum(neuronLayer[:layer])], 
             [-.5, nNeurons-.5], color='lightgrey', linestyle='-', linewidth=nNeurons/6)
    plt.plot([-.5, nNeurons-.5], [-.5+np.sum(neuronLayer[:layer]), -.5+np.sum(neuronLayer[:layer])], 
             color='lightgrey', linestyle='-', linewidth=nNeurons/6)
  plt.title('To', fontsize=15, y=1.05)
  plt.ylabel('From', fontsize=15)
  plt.show()

## ----------------------------------------------------------------------------
  
showConnection(connectionMatrix, adaptableMatrix, neuronLayer)
  
  
  
  
  
  
  