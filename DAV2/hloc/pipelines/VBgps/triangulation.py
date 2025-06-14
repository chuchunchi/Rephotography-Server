from pathlib import Path
import argparse

from ... import extract_features, match_features
from ... import pairs_from_poses, triangulation

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=Path, default='test',
                    help='Name of the scene under datasets dir, default: %(default)s')
args = parser.parse_args()

dataset = Path('datasets', args.name)
images = dataset / 'db_images/'

outputs = Path('outputs', args.name)
sfm_pairs = outputs / 'pairs-exhaustive.txt'  # exhaustive matching
sfm_dir = outputs / 'sfm_superpoint+NN'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['NN-superpoint']

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf['output'], outputs, exhaustive=True)

ref_sfm_model = dataset / 'ref_sfm_model/'

triangulation.main(sfm_dir, ref_sfm_model, images, sfm_pairs, feature_path, match_path)
