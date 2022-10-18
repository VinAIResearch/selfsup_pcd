import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
sys.path.append(os.path.join(BASE_DIR, "data_utils"))
