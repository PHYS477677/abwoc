import numpy as np

def generateImage(imSize,sunRadius,hasEvent,eventRadius,noiseLevel=None):
  """
  Generate an image of the Sun with or without a circular event. The Sun
  is given a constant value, with events and additional constant value on top.

  Inputs:
    imSize: int - size of (square) image in pixels. i.e., enter 4096 for 
            a 4096x4096 px image
    sunRadius: int - radius of Sun in pixels
    hasEvent: bool - whether to generate an event on the Sun
    eventRadius: int - radius of the event
    noise: float - standard deviation of Gaussian noise to add (mean 0).
            Experiment has shown that some level of noise is necessary to
            train a CNN on these generated images.

  Outputs:
    a 2d numpy array
  """

  #Create the Sun
  # xx and yy are tables containing the x and y coordinates as values
  # mgrid is a mesh creation helper
  xx, yy = np.mgrid[:imSize, :imSize]
  # circles contains the squared distance to the center point
  circle = (xx - imSize/2) ** 2 + (yy - imSize/2) ** 2
  # set sun to value of 0.2
  sun = np.float16((circle < (sunRadius**2))*0.2)

  if hasEvent:
    #Add an event
    #Generate location, make sure event is within sun
    eventMinLoc = imSize/2 - sunRadius/np.sqrt(2) + eventRadius
    eventx, eventy = np.random.randint(eventMinLoc, imSize-eventMinLoc, size=2)
    circle = (xx - eventx)**2 + (yy - eventy)**2
    event = np.float16((circle < (eventRadius**2)) * 0.6)
    sun += event

  if noiseLevel is not None:
    noise = np.random.normal(0, noiseLevel, size=(imSize,imSize))
    sun += np.float16(noise)

  return sun