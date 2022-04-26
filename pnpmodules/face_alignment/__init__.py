import sys
import os
import shutil
import glob
import platform
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'face_alignment'
sys.path.insert(0,str(module_path.resolve()))
sys.path.insert(0,str(Path(__file__).parent.resolve()))
#os.chdir(module_path)
#from face_alignment.api import FaceAlignment, LandmarksType, NetworkSize
#os.chdir(current_path)