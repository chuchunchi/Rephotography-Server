import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

def load_sg_model(conf):
    print(sys.path)
    path = Path(Path(__file__).parent, 'models', 'weights', 'model_epoch_51.pth')
    print('loading...')
    model = torch.load(path)

    return model


