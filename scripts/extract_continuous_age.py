"""Extract continuous ordinal age scores from the r34 age head for all FFHQ images.

AttributeClassifier contains a separate ResNet34 `age_heads` subnet with a
6-logit output. sigmoid(logits).sum(dim=1) gives a continuous ordinal score in
[0, 6] where higher = older. This avoids the binary CelebA Young label whose
old/young groups overlap heavily in the 35-45 year range.

Usage:
    python scripts/extract_continuous_age.py \
        --model_path data/r34_a40_age_256_classifier.pth \
        --img_dir data/FFHQ \
        --img_list data/ffhq.txt \
        --output data/ffhq_age_continuous.pth
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
        description='Extract continuous ordinal age scores via the r34 age_heads subnet.'
    )
    parser.add_argument('--model_path', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--img_dir', default='./data/FFHQ')
    parser.add_argument('--img_list', default='./data/ffhq.txt',
                        help='CSV with a "path" column (output of precompute_sdflow_data.py)')
    parser.add_argument('--output', default='./data/ffhq_age_continuous.pth')
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
    age_net = classifier.age_heads   # ResNet34 with fc = Linear(512, 6)
    age_net.eval().to(device)
    del classifier
    print('Loaded age_heads subnet (6-logit ordinal output, higher = older)')

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
    for batch in tqdm(loader, desc='age scores'):
        logits = age_net(batch.to(device))                # (B, 6)
        scores = torch.sigmoid(logits).sum(dim=1).cpu()  # (B,) in [0, 6]
        all_scores.append(scores)

    scores = torch.cat(all_scores)   # (N,)
    print(f'\nScore stats: min={scores.min():.3f}  max={scores.max():.3f}  '
          f'mean={scores.mean():.3f}  std={scores.std():.3f}')
    print('Percentile distribution:')
    for pct in [5, 10, 15, 25, 50, 75, 85, 90, 95]:
        v = torch.quantile(scores, pct / 100).item()
        n_young = (scores < v).sum().item() if pct <= 25 else None
        n_old   = (scores > v).sum().item() if pct >= 75 else None
        tag = f'  → {n_young} young samples below' if n_young else \
              f'  → {n_old} old samples above' if n_old else ''
        print(f'  p{pct:2d}: {v:.3f}{tag}')

    out = {
        'values': scores,
        'scale': '0-6_ordinal_higher_is_older',
        'source_img_list': str(args.img_list),
        'img_size': args.img_size,
    }
    torch.save(out, args.output)
    print(f'\nSaved → {args.output}  shape={tuple(scores.shape)}')


if __name__ == '__main__':
    main()
