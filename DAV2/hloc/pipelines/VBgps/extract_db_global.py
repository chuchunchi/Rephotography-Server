import argparse
from pathlib import Path

from hloc import extract_features

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=Path, default='test',
                    help='Name of the scene under datasets dir, default: %(default)s')
args = parser.parse_args()

dataset = Path('datasets', args.name)
images = dataset / 'db_images/'
outputs = Path('outputs', args.name)

feature_conf = extract_features.confs['netvlad']
feature_path = extract_features.main(feature_conf, images, outputs)
