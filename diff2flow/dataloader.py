import os
import torch
import numpy as np
import torchvision
import webdataset as wds
from einops import rearrange
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import ListConfig, DictConfig
from torch.utils.data import DataLoader, Dataset

from diff2flow.helpers import instantiate_from_config
from diff2flow.helpers import load_partial_from_config


""" WebDataset """


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


def identity(x):
    return x


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 tar_base,          # can be a list of paths or a single path
                 batch_size,
                 val_batch_size=None,
                 train=None,
                 validation=None,
                 test=None,
                 num_workers=4,
                 val_num_workers: int = None,
                 multinode=True,
                 remove_keys: list = None,          # list of keys to remove from the sample
                 ):
        super().__init__()
        if isinstance(tar_base, str):
            self.tar_base = tar_base
        elif isinstance(tar_base, ListConfig) or isinstance(tar_base, list):
            # check which tar_base exists
            for path in tar_base:
                if os.path.exists(path):
                    self.tar_base = path
                    break
            else:
                raise FileNotFoundError("Could not find a valid tarbase.")
        else:
            raise ValueError(f'Invalid tar_base type {type(tar_base)}')
        print(f'[WebDataModuleFromConfig] Setting tar base to {self.tar_base}')
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def make_loader(self, dataset_config, train=True):
        image_transforms = []
        lambda_fn = lambda x: x * 2. - 1.   # normalize to [-1, 1]
        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda_fn)])
        if 'image_transforms' in dataset_config:
            image_transforms.extend([instantiate_from_config(tt) for tt in dataset_config.image_transforms])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        # this is crucial to set correct image key to get the transofrms applied correctly
        img_keys = dataset_config.get('image_key', 'image.png')
        if isinstance(img_keys, str):
            img_keys = [img_keys]
        for img_key in img_keys:
            transform_dict.update({img_key: image_transforms})

        if 'dataset_transforms' in dataset_config:
            dataset_transforms = instantiate_from_config(dataset_config['dataset_transforms'])
        else:
            dataset_transforms = None

        if 'postprocess' in dataset_config:
            postprocess = instantiate_from_config(dataset_config['postprocess'])
        else:
            postprocess = None

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if isinstance(dataset_config.shards, str):
            tars = os.path.join(self.tar_base, dataset_config.shards)
        elif isinstance(dataset_config.shards, list) or isinstance(dataset_config.shards, ListConfig):
            # decompose into lists of shards
            # Turn train-{000000..000002}.tar into ['train-000000.tar', 'train-000001.tar', 'train-000002.tar']
            tars = []
            for shard in dataset_config.shards:
                # Assume that the shard starts from 000000
                if '{' in shard:
                    start, end = shard.split('..')
                    start = start.split('{')[-1]
                    end = end.split('}')[0]
                    start = int(start)
                    end = int(end)
                    tars.extend([shard.replace(f'{{{start:06d}..{end:06d}}}', f'{i:06d}') for i in range(start, end+1)])
                else:
                    tars.append(shard)
            tars = [os.path.join(self.tar_base, t) for t in tars]
            # random shuffle the shards
            if shardshuffle:
                np.random.shuffle(tars)
        else:
            raise ValueError(f'Invalid shards type {type(dataset_config.shards)}')

        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'[WebDataModuleFromConfig] Loading {len(dset.pipeline[0].urls):,} shards.')

        dset = (dset
                .decode('rgb', handler=wds.warn_and_continue)
                .map(self.filter_out_keys, handler=wds.warn_and_continue)
                .map_dict(**transform_dict, handler=wds.warn_and_continue)
                .map(self.to_float32, handler=wds.warn_and_continue)
                )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get('rename', None)
        if renaming is not None:
            dset = dset.rename(**renaming)

        if dataset_transforms is not None:
            dset = dset.map(dataset_transforms)

        if postprocess is not None:
            dset = dset.map(postprocess)
        
        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw)

        return loader

    def filter_out_keys(self, sample):
        for key in self.rm_keys:
            sample.pop(key, None)
        return sample
    
    def to_float32(self, sample):
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                sample[key] = sample[key].astype(np.float32)
            elif isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].float()
        return sample
    
    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)


""" Normal Dataset """


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 val_batch_size: int = None,
                 train: dict = None,
                 validation: dict = None,
                 test: dict = None,
                 shuffle_validation: bool = False,
                 num_workers: int = 0
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.validation = validation
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.shuffle_validation = shuffle_validation

        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle_validation)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle_validation)

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 10000000, **kwargs):
        """
        kwargs must contain the output keys with their corresponding shapes,
        e.g. image=(3, 32, 32), label=(1,)
        """
        super().__init__()
        self.num_samples = num_samples
        self.kwargs = {k: tuple(v) if not isinstance(v, str) else v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            k: v if isinstance(v, str) else torch.randn(v)
            for  k, v in self.kwargs.items()
        }


# --- 請將以下程式碼貼在 diff2flow/dataloader.py 的最下方 ---

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob

# --- 請將這段程式碼貼在 diff2flow/dataloader.py 的最下方 (覆蓋原本的 FolderDataset) ---

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class CelebAHQDataset(Dataset):
    def __init__(self, img_dir, attr_path, size=512):
        super().__init__()
        self.img_dir = img_dir
        self.size = size

        try:
            self.attr_df = pd.read_csv(attr_path, delim_whitespace=True, header=1)
        except:
            self.attr_df = pd.read_csv(attr_path)

        print(f"[CelebAHQ] Loaded {len(self.attr_df)} images with 40 attributes.")

        self.transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def attr_to_text(self, row):
        """
        將 CelebA 的 40 個屬性完整轉換為自然語言描述
        """
        # --- [1, 2] 基礎身分 ---
        # 1. Male, 2. Young
        gender = "man" if row.get('Male', -1) == 1 else "woman"  # [No.21] Male
        age = "young" if row.get('Young', -1) == 1 else "old"  # [No.40] Young
        text = f"a photo of a {age} {gender}"

        traits = []

        # --- [3-9] 整體狀態與膚質 ---
        if row.get('Smiling', -1) == 1: traits.append("smiling")  # [No.32] Smiling
        if row.get('Attractive', -1) == 1: traits.append("attractive")  # [No.3] Attractive
        if row.get('Blurry', -1) == 1: traits.append("blurry")  # [No.11] Blurry
        if row.get('Chubby', -1) == 1: traits.append("chubby")  # [No.14] Chubby
        if row.get('Pale_Skin', -1) == 1: traits.append("with pale skin")  # [No.27] Pale_Skin
        if row.get('Oval_Face', -1) == 1: traits.append("with oval face")  # [No.26] Oval_Face
        if row.get('Heavy_Makeup', -1) == 1: traits.append("wearing heavy makeup")  # [No.19] Heavy_Makeup

        # --- [10-18] 髮型與髮色 ---
        hair_traits = []
        if row.get('Bald', -1) == 1: hair_traits.append("bald head")  # [No.5] Bald
        if row.get('Black_Hair', -1) == 1: hair_traits.append("black hair")  # [No.9] Black_Hair
        if row.get('Blond_Hair', -1) == 1: hair_traits.append("blond hair")  # [No.10] Blond_Hair
        if row.get('Brown_Hair', -1) == 1: hair_traits.append("brown hair")  # [No.12] Brown_Hair
        if row.get('Gray_Hair', -1) == 1: hair_traits.append("gray hair")  # [No.18] Gray_Hair
        if row.get('Receding_Hairline', -1) == 1: hair_traits.append("receding hairline")  # [No.29] Receding_Hairline
        if row.get('Straight_Hair', -1) == 1: hair_traits.append("straight hair")  # [No.33] Straight_Hair
        if row.get('Wavy_Hair', -1) == 1: hair_traits.append("wavy hair")  # [No.34] Wavy_Hair
        if row.get('Bangs', -1) == 1: hair_traits.append("bangs")  # [No.6] Bangs

        if hair_traits:
            traits.append("with " + " and ".join(hair_traits))

        # --- [19-24] 鬍鬚狀態 ---
        beard_traits = []
        # [No.25] No_Beard (這個屬性是反義，我們通常描述"有什麼"，但如果特別強調沒鬍子可加 clean-shaven)
        if row.get('No_Beard', -1) == 1 and row.get('Male', -1) == 1:
            beard_traits.append("clean-shaven face")

        if row.get('5_o_Clock_Shadow', -1) == 1: beard_traits.append("5 o'clock shadow")  # [No.1] 5_o_Clock_Shadow
        if row.get('Goatee', -1) == 1: beard_traits.append("goatee")  # [No.17] Goatee
        if row.get('Mustache', -1) == 1: beard_traits.append("mustache")  # [No.23] Mustache
        if row.get('Sideburns', -1) == 1: beard_traits.append("sideburns")  # [No.31] Sideburns

        if beard_traits:
            traits.append("with " + " and ".join(beard_traits))

        # --- [25-35] 五官特徵 ---
        face_features = []
        if row.get('Arched_Eyebrows', -1) == 1: face_features.append("arched eyebrows")  # [No.2] Arched_Eyebrows
        if row.get('Bushy_Eyebrows', -1) == 1: face_features.append("bushy eyebrows")  # [No.13] Bushy_Eyebrows
        if row.get('Bags_Under_Eyes', -1) == 1: face_features.append("bags under eyes")  # [No.4] Bags_Under_Eyes
        if row.get('Narrow_Eyes', -1) == 1: face_features.append("narrow eyes")  # [No.24] Narrow_Eyes
        if row.get('Big_Nose', -1) == 1: face_features.append("big nose")  # [No.8] Big_Nose
        if row.get('Pointy_Nose', -1) == 1: face_features.append("pointy nose")  # [No.28] Pointy_Nose
        if row.get('Big_Lips', -1) == 1: face_features.append("big lips")  # [No.7] Big_Lips
        if row.get('Mouth_Slightly_Open', -1) == 1: face_features.append(
            "mouth slightly open")  # [No.22] Mouth_Slightly_Open
        if row.get('High_Cheekbones', -1) == 1: face_features.append("high cheekbones")  # [No.20] High_Cheekbones
        if row.get('Rosy_Cheeks', -1) == 1: face_features.append("rosy cheeks")  # [No.30] Rosy_Cheeks
        if row.get('Double_Chin', -1) == 1: face_features.append("double chin")  # [No.15] Double_Chin

        if face_features:
            traits.append("with " + ", ".join(face_features))

        # --- [36-40] 配件與穿著 ---
        accessories = []
        if row.get('Eyeglasses', -1) == 1: accessories.append("eyeglasses")  # [No.16] Eyeglasses
        if row.get('Wearing_Earrings', -1) == 1: accessories.append("earrings")  # [No.35] Wearing_Earrings
        if row.get('Wearing_Hat', -1) == 1: accessories.append("a hat")  # [No.36] Wearing_Hat
        if row.get('Wearing_Lipstick', -1) == 1: accessories.append("lipstick")  # [No.37] Wearing_Lipstick
        if row.get('Wearing_Necklace', -1) == 1: accessories.append("a necklace")  # [No.38] Wearing_Necklace
        if row.get('Wearing_Necktie', -1) == 1: accessories.append("a necktie")  # [No.39] Wearing_Necktie

        if accessories:
            traits.append("wearing " + " and ".join(accessories))

        # 組合所有描述
        if traits:
            text += ", " + ", ".join(traits)

        return text

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        fname = self.attr_df.index[idx]
        if not str(fname).endswith(".jpg"):
            fname = f"{fname}.jpg"

        img_path = os.path.join(self.img_dir, fname)

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            caption = self.attr_to_text(self.attr_df.iloc[idx])

            return {
                "x1": img_tensor,
                "x0": img_tensor,
                "txt": caption
            }
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

if __name__ == "__main__":
    # TODO: implement small test
    pass
