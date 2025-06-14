from pathlib import Path
import argparse

from ... import extract_features, match_features
from ... import pairs_from_poses, triangulation

parser = argparse.ArgumentParser()
parser.add_argument('--sfm_dir', type=Path, required=True)
parser.add_argument('--reference_sfm_model', type=Path, required=True)
parser.add_argument('--image_dir', type=Path, required=True)

parser.add_argument('--pairs', type=Path, required=True)
parser.add_argument('--features', type=Path, required=True)
parser.add_argument('--matches', type=Path, required=True)

parser.add_argument('--colmap_path', type=Path, default='colmap')

parser.add_argument('--skip_geometric_verification', action='store_true')
parser.add_argument('--min_match_score', type=float)
args = parser.parse_args()

triangulation.main(**args.__dict__)
