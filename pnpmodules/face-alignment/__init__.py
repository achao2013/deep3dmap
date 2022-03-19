import sys
import os
import shutil
import glob
import platform
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'face_alignment'
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)
from api import FaceAlignment, LandmarksType, NetworkSize
os.chdir(current_path)