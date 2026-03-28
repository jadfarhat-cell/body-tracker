"""
Realistic Face & Body Swap with Expression Mirroring
The swapped face mirrors your expressions - blinks, smiles, etc.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import numpy as np
import sys
