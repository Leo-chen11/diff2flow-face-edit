import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.e4e import Encoder4Editing


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tiff'}


class ImagePathDataset(Dataset):
    def __init__(self, image_root, paths, img_size):
        self.image_root = Path(image_root)
        self.paths = list(paths)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rel_path = self.paths[index]
        image = Image.open(self.image_root / rel_path).convert('RGB')
        return self.transform(image), rel_path


def find_images(image_root, recursive):
    image_root = Path(image_root)
    pattern = '**/*' if recursive else '*'
    paths = [
        path.relative_to(image_root).as_posix()
        for path in image_root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    ]
    return sorted(paths)


def write_index_file(paths, output_path, test_ratio, seed):
    rng = random.Random(seed)
    test_count = int(len(paths) * test_ratio)
    test_paths = set(rng.sample(paths, test_count)) if test_count > 0 else set()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'split'])
        for path in paths:
            writer.writerow([path, 1 if path in test_paths else 0])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='Precompute SDFlow training files: ffhq.txt, e4e latents, and attribute predictions.'
    )
    parser.add_argument('--image_root', default='./data/FFHQ', help='folder containing aligned face images')
    parser.add_argument('--index_file', default='./data/ffhq.txt')
    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--e4e_weights', default='./data/e4e_ffhq_encode.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--recursive', action='store_true', help='scan images recursively under image_root')
    args = parser.parse_args()

    image_root = Path(args.image_root)
    paths = find_images(image_root, args.recursive)
    if not paths:
        raise RuntimeError(f'No images found under {image_root}')

    write_index_file(paths, args.index_file, args.test_ratio, args.seed)
    print(f'Wrote {args.index_file} with {len(paths)} images')

    dataset = ImagePathDataset(image_root, paths, args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith('cuda'),
    )

    device = torch.device(args.device)
    e4e = Encoder4Editing(
        num_layers=50,
        mode='ir_se',
        stylegan_size=1024,
        checkpoint_path=args.e4e_weights,
    ).to(device).eval()
    ckpt = torch.load(args.e4e_weights, map_location='cpu')
    latent_avg = ckpt['latent_avg'].to(device)
    print('Loaded e4e encoder')

    classifier = AttributeClassifier(backbone='r34')
    classifier.load_state_dict(load_network(args.attribute_weights))
    classifier = classifier.to(device).eval()
    print('Loaded attribute classifier')

    all_paths = []
    all_latents = []
    all_preds = []
    for images, rel_paths in tqdm(loader, total=len(loader)):
        images = images.to(device, non_blocking=True)
        latents = e4e(images) + latent_avg
        _, preds = classifier(images)

        all_paths.extend(rel_paths)
        all_latents.append(latents.cpu())
        all_preds.append(preds.cpu())

    latent_payload = {'paths': all_paths, 'values': torch.cat(all_latents, dim=0)}
    preds_payload = {'paths': all_paths, 'values': torch.cat(all_preds, dim=0)}

    Path(args.latent_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.preds_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent_payload, args.latent_file)
    torch.save(preds_payload, args.preds_file)

    print(f'Wrote {args.latent_file}: {tuple(latent_payload["values"].shape)}')
    print(f'Wrote {args.preds_file}: {tuple(preds_payload["values"].shape)}')


if __name__ == '__main__':
    main()
