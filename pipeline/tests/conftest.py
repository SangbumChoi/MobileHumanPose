import os.path as osp
import sys

# Make the pipeline package importable (common, stageXX, inference, ...).
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
