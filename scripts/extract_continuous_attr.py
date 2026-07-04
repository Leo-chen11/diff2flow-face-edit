"""Extract continuous (pre-threshold) sigmoid scores for the 40 CelebA
binary attribute heads, generalizing extract_continuous_age.py so that every
attribute -- not just age -- has a continuous confidence score available for
percentile-extreme splits when computing direction-bank directions (see
scripts/precompute_directions_stratified.py).

tools/precompute_sdflow_data.py only saves AttributeClassifier's already
thresholded 0/1 predictions (sigmoid(logits) > 0.5), so there is no continuous
signal to select confidently-labeled samples from. AttributeClassifier.
forward_attr returns the raw logits for the 40 binary heads before
thresholding; sigmoid(logits) gives a continuous [0, 1] score per attribute.

Usage:
    python scripts/extract_continuous_attrs.py \
        --model_path data/r34_a40_age_256_classifier.pth \
        --img_dir data/FFHQ \
        --img_list data/ffhq.txt \
        --output data/ffhq_e4e_preds_continuous.pth
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.attribute_estimator import AttributeClassifier


class _ImageDataset(Dataset):
    def __init__(self, image_root, paths, img_size=256):
        self.image_root = Path(image_root)
        self.paths = paths
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_root / self.paths[idx]).convert('RGB')
        return self.transform(img)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Extract continuous (pre-threshold) sigmoid scores for the 40 CelebA attribute heads.'
    )
    parser.add_argument('--model_path', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--img_dir', default='./data/FFHQ')
    parser.add_argument('--img_list', default='./data/ffhq.txt',
                        help='CSV with a "path" column (output of precompute_sdflow_data.py)')
    parser.add_argument('--output', default='./data/ffhq_e4e_preds_continuous.pth')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input resolution (matches the "256" in the model filename)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading model from {args.model_path} ...')
    classifier = AttributeClassifier()
    classifier.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    classifier.eval().to(device)
    print('Loaded AttributeClassifier (forward_attr: 40 binary attribute heads, pre-threshold)')

    with open(args.img_list, newline='') as f:
        reader = csv.DictReader(f)
        paths = [row['path'] for row in reader]
    print(f'Images: {len(paths)} from {args.img_list}')

    dataset = _ImageDataset(args.img_dir, paths, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        shuffle=False,
    )

    all_scores = []
    for batch in tqdm(loader, desc='continuous attr scores'):
        logits, _ = classifier.forward_attr(batch.to(device))   # (B, 40)
        all_scores.append(torch.sigmoid(logits).cpu())

    scores = torch.cat(all_scores)   # (N, 40)
    print(f'\nShape: {tuple(scores.shape)}')
    for idx, name in [(15, 'Eyeglasses'), (20, 'Male'), (39, 'Young')]:
        col = scores[:, idx]
        print(f'  attr {idx:>2} ({name:<10}): mean={col.mean():.3f} std={col.std():.3f} '
              f'min={col.min():.3f} max={col.max():.3f}')

    out = {'paths': paths, 'values': scores}
    torch.save(out, args.output)
    print(f'\nSaved -> {args.output}  shape={tuple(scores.shape)}')


if __name__ == '__main__':
    main()
