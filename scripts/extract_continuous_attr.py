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

IS THIS SCORE ACTUALLY TRUSTWORTHY? The general idea -- use a pretrained
classifier's confidence to pick the most confidently-labeled samples for
direction extraction -- is standard (this is what InterFaceGAN and similar
linear-direction methods do with an SVM's decision-boundary distance). But
it comes with two real caveats, both already documented elsewhere in this
project:
  1. This is the SAME r34 classifier used as the frozen training-time
     "teacher" loss. Any systematic blind spot it has propagates into BOTH
     -- the direction geometry inherits exactly the same failure mode the
     training loss can be gamed through. This project's own eval has
     measured a large teacher-vs-independent-judge gap specifically on
     eyeglasses (r34 ~91% vs CLIP zero-shot judge far lower) -- so a sample
     this classifier is "confidently" eyeglasses-positive is not
     automatically one a human, or an architecturally different judge,
     would agree with.
  2. Modern deep classifiers are well known to be poorly calibrated (Guo et
     al. 2017, "On Calibration of Modern Neural Networks") -- raw
     sigmoid/softmax outputs tend toward overconfidence, clustering near
     0/1 even on genuinely ambiguous inputs. A percentile split on this raw
     score selects samples the network is confident ABOUT, not necessarily
     samples it is confidently CORRECT about.

--cross_judge clip cross-checks against an architecturally independent
judge (OpenAI CLIP zero-shot, same class evaluate_sdflow.py uses for its
headline AccCLIP column) on the SAME images, and reports correlation +
disagreement rate per attribute -- an empirical answer to "how much should
I trust this score" instead of assuming the InterFaceGAN precedent
transfers cleanly. The CLIP scores are saved alongside the r34 ones so
scripts/precompute_directions_stratified.py can optionally require both
judges to agree before counting a sample as confidently high/low
(--require_cross_judge_agree).

Usage:
    python scripts/extract_continuous_attrs.py \
        --model_path data/r34_a40_age_256_classifier.pth \
        --img_dir data/FFHQ \
        --img_list data/ffhq.txt \
        --output data/ffhq_e4e_preds_continuous.pth \
        --cross_judge clip
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    parser.add_argument('--cross_judge', default='none', choices=['none', 'clip'],
                        help="'clip' additionally scores every image with an independent "
                             "CLIP zero-shot judge (same one evaluate_sdflow.py uses) and "
                             "reports correlation/disagreement against the r34 continuous "
                             "score, per attribute in --cross_judge_attrs -- see module "
                             "docstring for why this matters. 'none' (default) reproduces "
                             "the old r34-only behavior.")
    parser.add_argument('--cross_judge_model', default='ViT-L/14',
                        help='CLIP model for --cross_judge clip. Deliberately different from '
                             'the ViT-B/32 used elsewhere in this project (conditioner/clip '
                             'prompt loss) to stay an independent check.')
    parser.add_argument('--cross_judge_attrs', nargs='*', type=int, default=[15, 20, 39],
                        help='Which attribute indices to cross-check (CLIPAttributeJudge has '
                             'hand-written prompts for 15/20/24/31/33/39; others fall back to '
                             'a generic prompt pair and are less meaningful to cross-check).')
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

    clip_judge = None
    if args.cross_judge == 'clip':
        from evaluation.evaluate_sdflow import CLIPAttributeJudge
        clip_judge = CLIPAttributeJudge(args.cross_judge_attrs, args.cross_judge_model, device)
        print(f'[CrossJudge] CLIP zero-shot ({args.cross_judge_model}) scoring attrs '
              f'{args.cross_judge_attrs} alongside the r34 continuous scores.')

    all_scores = []
    all_cross = [] if clip_judge is not None else None
    for batch in tqdm(loader, desc='continuous attr scores'):
        batch = batch.to(device)
        logits, _ = classifier.forward_attr(batch)   # (B, 40)
        all_scores.append(torch.sigmoid(logits).cpu())
        if clip_judge is not None:
            all_cross.append(clip_judge.scores(batch).cpu())   # (B, len(cross_judge_attrs))

    scores = torch.cat(all_scores)   # (N, 40)
    print(f'\nShape: {tuple(scores.shape)}')
    for idx, name in [(15, 'Eyeglasses'), (20, 'Male'), (39, 'Young')]:
        col = scores[:, idx]
        print(f'  attr {idx:>2} ({name:<10}): mean={col.mean():.3f} std={col.std():.3f} '
              f'min={col.min():.3f} max={col.max():.3f}')

    out = {'paths': paths, 'values': scores}

    if clip_judge is not None:
        cross_scores = torch.cat(all_cross)   # (N, len(cross_judge_attrs))
        out['values_cross_clip'] = cross_scores
        out['cross_attribute_index'] = args.cross_judge_attrs
        attr_names = {15: 'Eyeglasses', 20: 'Male', 24: 'No_Beard', 31: 'Smiling',
                     33: 'Wavy_Hair', 39: 'Young'}
        print('\n=== r34 (teacher) vs CLIP zero-shot (independent) agreement ===')
        print('How to read this: high correlation + low disagreement means the r34 continuous '
              'score is a trustworthy confidence signal for THIS attribute -- percentile splits '
              'on it (--extreme_pct) are picking samples an independent judge agrees are '
              'extreme. Low correlation / high disagreement means the two are seeing different '
              'things -- --extreme_pct is then selecting samples that are merely "confident to '
              'r34", not confidently-and-verifiably extreme, and --require_cross_judge_agree '
              '(precompute_directions_stratified.py) is worth turning on for that attribute.')
        for i, idx in enumerate(args.cross_judge_attrs):
            r34_col = scores[:, idx]
            clip_col = cross_scores[:, i]
            corr = torch.corrcoef(torch.stack([r34_col, clip_col]))[0, 1].item()
            r34_side = r34_col >= 0.5
            clip_side = clip_col >= 0.5
            disagree = (r34_side != clip_side).float().mean().item()
            print(f'  attr {idx:>2} ({attr_names.get(idx, "?"):<10}): '
                  f'pearson_r={corr:6.3f}  disagree(0.5 side)={disagree*100:5.1f}%')

    torch.save(out, args.output)
    print(f'\nSaved -> {args.output}  shape={tuple(scores.shape)}'
          + (f'  (+ values_cross_clip {tuple(cross_scores.shape)})' if clip_judge is not None else ''))


if __name__ == '__main__':
    main()
