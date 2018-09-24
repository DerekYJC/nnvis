'''
Define the connection function to connect the neurons ...
  Example of connectionMatrix:
    connectionMatrix[from_neuron][to_neuron] = weight 
'''
connectionMatrix = np.zeros([nNeurons, nNeurons])
adaptableMatrix  = np.zeros([nNeurons, nNeurons])
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
