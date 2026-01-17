"""
編輯微調腳本 - 完全修正版 v3
主要修正：
1. 增強 Prompt 差異（提高 drop_ratio，增加更多屬性）
2. 修正 v_target 計算（使用正確的編輯目標）
3. 增加訓練監控和診斷
4. 優化 learning rate 和訓練策略
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
import os
import random
import pandas as pd

from diff2flow.trainer_module import TrainerModuleLatentFM


class PseudoEditingDataset(Dataset):
    """Pseudo-Editing Dataset（強化版）"""

    def __init__(self, data_root, attributes_file=None, noise_level=0.3, drop_ratio=0.5):
        self.data_root = data_root
        self.noise_level = noise_level
        self.drop_ratio = drop_ratio  # 新增：可調整的 drop_ratio
        self.transform = self.get_transform()

        # 自動尋找 attributes.txt
        if attributes_file is None:
            auto_attr = os.path.join(data_root, "attributes.txt")
            if os.path.exists(auto_attr):
                attributes_file = auto_attr
                print(f"✓ 自動找到屬性檔案: {auto_attr}")

        # 載入資料
        self.images = self.load_images()

        # 載入屬性
        if attributes_file and os.path.exists(attributes_file):
            self.attributes = self.load_attributes(attributes_file)
            if self.attributes:
                print(f"✓ 屬性檔案載入成功，將使用真實屬性")
            else:
                print(f"⚠️  屬性檔案載入失敗，將使用預設屬性")
        else:
            print("⚠️  未提供屬性檔案，將使用預設 prompts")
            self.attributes = None

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load_images(self):
        """使用 os.listdir（支援中文路徑）"""
        image_dir = os.path.join(self.data_root, "images")
        if not os.path.exists(image_dir):
            print(f"⚠️  images 子資料夾不存在，使用根目錄: {self.data_root}")
            image_dir = self.data_root
        else:
            print(f"✓ 找到 images 資料夾: {image_dir}")

        valid_extensions = (
            '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
            '.bmp', '.BMP', '.webp', '.WEBP'
        )

        images = []

        try:
            print(f"🔍 搜尋圖片在: {image_dir}")
            all_files = os.listdir(image_dir)

            for filename in all_files:
                if filename.endswith(valid_extensions):
                    full_path = os.path.join(image_dir, filename)
                    if os.path.isfile(full_path):
                        images.append(full_path)

            if len(images) == 0:
                print(f"⚠️  在 {image_dir} 找不到圖片，嘗試搜尋子資料夾...")
                for root, dirs, files in os.walk(image_dir):
                    for filename in files:
                        if filename.endswith(valid_extensions):
                            full_path = os.path.join(root, filename)
                            images.append(full_path)

        except Exception as e:
            print(f"❌ 錯誤：無法讀取目錄: {e}")
            return []

        if len(images) == 0:
            print(f"❌ 錯誤：找不到任何圖片！")
        else:
            print(f"✓ 總共找到 {len(images)} 張圖片")
            for img in images[:3]:
                print(f"   範例: {os.path.basename(img)}")

        return sorted(images)

    def load_attributes(self, attributes_file):
        """載入屬性檔案並建立映射"""
        print(f"載入屬性檔案: {attributes_file}")

        try:
            with open(attributes_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.isdigit():
                    skiprows = 1
                else:
                    skiprows = 0

            df = pd.read_csv(attributes_file, delim_whitespace=True, skiprows=skiprows)

            attr_dict = {}
            for idx, row in df.iterrows():
                image_id = str(row.iloc[0])
                image_id_base = os.path.splitext(image_id)[0]

                attrs = {}
                for col in df.columns[1:]:
                    attrs[col] = 1 if row[col] == 1 else 0

                # 儲存多種格式
                attr_dict[image_id] = attrs
                attr_dict[image_id_base] = attrs

                if image_id_base.isdigit():
                    attr_dict[f"{int(image_id_base):05d}"] = attrs

            print(f"✓ 載入了 {len(df)} 個圖片的屬性（{len(df.columns)-1} 個屬性欄位）")
            return attr_dict

        except Exception as e:
            print(f"⚠️  載入屬性檔案失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_image_attributes(self, image_path):
        if self.attributes is None:
            return self.get_default_attributes()

        filename = os.path.basename(image_path)
        image_id = os.path.splitext(filename)[0]

        possible_ids = [
            image_id,
            filename,
        ]

        if image_id.isdigit():
            possible_ids.append(f"{int(image_id):05d}")

        for pid in possible_ids:
            if pid in self.attributes:
                return self.attributes[pid]

        return self.get_default_attributes()

    def get_default_attributes(self):
        """預設屬性（當找不到時）"""
        return {
            'Male': 0, 'Young': 1, 'Smiling': 1, 'Eyeglasses': 0,
            'Black_Hair': 1, 'Blond_Hair': 0, 'Brown_Hair': 0,
            'Wearing_Hat': 0, 'Wearing_Earrings': 0, 'Heavy_Makeup': 0
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        attributes = self.get_image_attributes(image_path)

        # Source: 加雜訊
        noise = torch.randn_like(image_tensor) * self.noise_level
        source_tensor = torch.clamp(image_tensor + noise, -1, 1)

        # Target: 原圖
        target_tensor = image_tensor

        # Prompts: Source 使用高 drop_ratio，Target 使用 0
        source_prompt = self.generate_prompt(attributes, drop_ratio=self.drop_ratio)
        target_prompt = self.generate_prompt(attributes, drop_ratio=0.0)

        return {
            'source_image': source_tensor,
            'target_image': target_tensor,
            'source_prompt': source_prompt,
            'target_prompt': target_prompt
        }

    def generate_prompt(self, attributes, drop_ratio=0.0):
        """
        生成 prompt（強化版 - 包含更多屬性）

        drop_ratio: 丟棄屬性的機率
        - 0.0: 保留所有屬性（Target）
        - 0.5: 丟棄 50% 屬性（Source）
        """
        # 基礎描述
        gender = 'man' if attributes.get('Male', 0) == 1 else 'woman'
        age = 'young' if attributes.get('Young', 1) == 1 else 'old'
        prompt = f"a photo of a {age} {gender}"

        features = []

        # 1. 髮色（重要！互斥屬性）
        hair_attrs = [
            ('Black_Hair', 'black hair'),
            ('Blond_Hair', 'blond hair'),
            ('Brown_Hair', 'brown hair'),
            ('Gray_Hair', 'gray hair')
        ]
        for attr, desc in hair_attrs:
            if attributes.get(attr, 0) == 1:
                if random.random() > drop_ratio:
                    features.append(desc)
                break  # 只取一種髮色

        # 2. 臉部特徵（重要！）
        facial_features = [
            ('Smiling', 'smiling'),
            ('Mouth_Slightly_Open', 'mouth slightly open'),
            ('High_Cheekbones', 'high cheekbones'),
            ('Big_Nose', 'big nose'),
            ('Pointy_Nose', 'pointy nose'),
            ('Chubby', 'chubby face'),
            ('Oval_Face', 'oval face')
        ]
        for attr, desc in facial_features:
            if attributes.get(attr, 0) == 1:
                if random.random() > drop_ratio:
                    features.append(desc)

        # 3. 化妝和皮膚
        makeup_features = [
            ('Heavy_Makeup', 'wearing heavy makeup'),
            ('Wearing_Lipstick', 'wearing lipstick'),
            ('Rosy_Cheeks', 'rosy cheeks'),
            ('Pale_Skin', 'pale skin')
        ]
        for attr, desc in makeup_features:
            if attributes.get(attr, 0) == 1:
                if random.random() > drop_ratio:
                    features.append(desc)

        # 4. 配件（最重要！編輯效果明顯）
        accessories = []
        accessory_attrs = [
            ('Eyeglasses', 'eyeglasses'),
            ('Wearing_Hat', 'hat'),
            ('Wearing_Earrings', 'earrings'),
            ('Wearing_Necklace', 'necklace'),
            ('Wearing_Necktie', 'necktie')
        ]
        for attr, desc in accessory_attrs:
            if attributes.get(attr, 0) == 1:
                if random.random() > drop_ratio:
                    accessories.append(desc)

        # 5. 鬍子相關（男性）
        if attributes.get('Male', 0) == 1:
            beard_features = [
                ('Mustache', 'mustache'),
                ('Goatee', 'goatee'),
                ('Sideburns', 'sideburns'),
                ('5_o_Clock_Shadow', '5 o\'clock shadow')
            ]
            for attr, desc in beard_features:
                if attributes.get(attr, 0) == 1:
                    if random.random() > drop_ratio:
                        features.append(desc)

        # 組合特徵
        if features:
            prompt += ", " + ", ".join(features)

        # 組合配件
        if accessories:
            prompt += ", wearing " + " and ".join(accessories)

        return prompt


class EditingFineTuner(pl.LightningModule):
    """編輯微調模組（強化版）"""

    def __init__(self, pretrained_ckpt, config, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        print(f"正在載入 Checkpoint: {pretrained_ckpt}")
        self.model = TrainerModuleLatentFM.load_from_checkpoint(
            pretrained_ckpt,
            map_location='cpu',
            **self.get_model_kwargs(config)
        )

        # 自動偵測 Text Encoder
        self.text_encoder = self._find_text_encoder()

        self.learning_rate = learning_rate
        self.automatic_optimization = False

        # 記錄統計
        self.v_norm_history = []
        self.prompt_same_count = 0
        self.prompt_total_count = 0

    def _find_text_encoder(self):
        """自動偵測 Text Encoder"""
        print("🔍 正在偵測 Text Encoder...")

        if hasattr(self.model, 'cond_stage'):
            print(f"✅ 找到 self.model.cond_stage")
            return self.model.cond_stage

        candidates = ['cond_stage_model', 'conditioner', 'text_encoder']
        for name in candidates:
            if hasattr(self.model, name):
                print(f"✅ 找到 self.model.{name}")
                return getattr(self.model, name)

        if hasattr(self.model, 'model'):
            for name in ['cond_stage', 'cond_stage_model']:
                if hasattr(self.model.model, name):
                    print(f"✅ 找到 self.model.model.{name}")
                    return getattr(self.model.model, name)

        print("❌ 警告：無法自動偵測 Text Encoder")
        return None

    def encode_text(self, text_list):
        """統一的文字編碼介面"""
        if self.text_encoder is None:
            raise RuntimeError("Text Encoder 未找到！")

        if hasattr(self.text_encoder, 'encode'):
            return self.text_encoder.encode(text_list)

        return self.text_encoder(text_list)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        source_img = batch['source_image']
        target_img = batch['target_image']
        source_prompt = batch['source_prompt']
        target_prompt = batch['target_prompt']

        # 統計 Prompt 相同率
        for sp, tp in zip(source_prompt, target_prompt):
            self.prompt_total_count += 1
            if sp == tp:
                self.prompt_same_count += 1

        # 每 100 步輸出統計
        if batch_idx % 100 == 0 and self.prompt_total_count > 0:
            same_ratio = self.prompt_same_count / self.prompt_total_count
            print(f"\n[Step {batch_idx}] Prompt 相同率: {same_ratio*100:.1f}%")
            if same_ratio > 0.3:
                print(f"⚠️  警告：Prompt 相同率過高！應該接近 0%")

        with torch.no_grad():
            z_source = self.model.encode_first_stage(source_img)
            z_target = self.model.encode_first_stage(target_img)
            c_target = self.encode_text(target_prompt)

        # Flow Matching
        t = torch.rand(z_source.shape[0], device=self.device)

        # ⚠️ 關鍵：這裡是編輯的核心
        # z_t 是從 source 到 target 的插值
        z_t = (1 - t[:, None, None, None]) * z_source + t[:, None, None, None] * z_target

        # v_target 是「從 source 到 target」的速度場
        v_target = z_target - z_source

        # 模型預測：給定當前狀態 z_t 和目標 prompt，預測應該往哪個方向走
        v_pred = self.model.model(
            z_t,
            t,
            context=z_source,      # Identity preservation
            context_ca=c_target    # Target prompt guidance
        )

        # Loss
        loss = F.mse_loss(v_pred, v_target)

        # 優化
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        opt.step()

        # 記錄
        v_norm = v_pred.norm().item()
        self.v_norm_history.append(v_norm)

        # 每 50 步計算平均
        if batch_idx % 50 == 0 and len(self.v_norm_history) > 0:
            avg_v_norm = sum(self.v_norm_history[-50:]) / min(50, len(self.v_norm_history))
            print(f"[Step {batch_idx}] avg_v_norm (last 50): {avg_v_norm:.2f}")

            if avg_v_norm > 50:
                print(f"⚠️  v_norm 偏高！")
            elif avg_v_norm < 15:
                print(f"✓ v_norm 良好")

        self.log('train_loss', loss, prog_bar=True)
        self.log('v_norm', v_norm, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7
        )

        return [optimizer], [scheduler]

    @staticmethod
    def get_model_kwargs(config):
        return {
            'fm_cfg': config.model.fm_cfg,
            'first_stage': config.autoencoder,
            'cond_stage_cfg': config.task.cond_stage_cfg,
            'lora_cfg': config.lora.lora_cfg,
            'start_from_noise': config.model.start_from_noise,
            'noising_step': config.model.noising_step,
            'scale_factor': config.autoencoder.get('scale_factor', 0.18215)
        }


def train_editing_finetuner(
    pretrained_ckpt,
    config_path,
    train_data_root,
    val_data_root=None,
    train_attributes_file=None,
    val_attributes_file=None,
    output_dir="./editing_finetuned",
    use_combined_data=False,
    max_epochs=30,
    batch_size=4,
    learning_rate=1e-5,
    noise_level=0.3,
    drop_ratio=0.5  # 新增：可調整的 drop_ratio
):
    """執行編輯微調"""

    config = OmegaConf.load(config_path)

    print("="*60)
    print("訓練配置")
    print("="*60)
    print(f"  Noise Level: {noise_level}")
    print(f"  Drop Ratio: {drop_ratio}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Epochs: {max_epochs}")
    print("="*60 + "\n")

    print("使用 Pseudo-Editing 訓練...")
    train_dataset = PseudoEditingDataset(
        train_data_root,
        train_attributes_file,
        noise_level=noise_level,
        drop_ratio=drop_ratio
    )

    print(f"訓練資料: {len(train_dataset)} 張")

    if len(train_dataset) == 0:
        print("❌ 訓練資料為空，終止執行")
        return

    # 測試前 3 個樣本
    print("\n" + "="*60)
    print("測試前 3 個訓練樣本")
    print("="*60)
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"\n[樣本 {i+1}]")
        print(f"  Source: {sample['source_prompt']}")
        print(f"  Target: {sample['target_prompt']}")
        print(f"  相同: {'❌ 有問題！' if sample['source_prompt'] == sample['target_prompt'] else '✓'}")
    print("="*60 + "\n")

    val_dataset = None
    if use_combined_data:
        print("⚠️  使用合併模式：train+val 一起訓練")
        if val_data_root:
            val_part = PseudoEditingDataset(
                val_data_root,
                val_attributes_file,
                noise_level=noise_level,
                drop_ratio=drop_ratio
            )
            train_dataset = ConcatDataset([train_dataset, val_part])
            print(f"合併後訓練資料: {len(train_dataset)} 張")
    elif val_data_root:
        print(f"驗證資料: {val_data_root}")
        val_dataset = PseudoEditingDataset(
            val_data_root,
            val_attributes_file,
            noise_level=noise_level,
            drop_ratio=drop_ratio
        )
        print(f"驗證資料: {len(val_dataset)} 張")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloader = None
    if val_dataset and len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    model = EditingFineTuner(
        pretrained_ckpt=pretrained_ckpt,
        config=config,
        learning_rate=learning_rate
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        default_root_dir=output_dir,
        log_every_n_steps=10,
        val_check_interval=0.5 if val_dataloader else None,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{output_dir}/checkpoints",
                filename='editing-{epoch:02d}-{train_loss:.4f}',
                save_top_k=3,
                monitor='train_loss',
                mode='min',
                save_last=True,
                every_n_epochs=5
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    print("\n🚀 開始微調訓練...")
    if val_dataloader:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader)

    print(f"\n✓ 訓練完成！")
    print(f"Checkpoints 儲存在: {output_dir}/checkpoints/")


if __name__ == "__main__":
    # 設定
    PRETRAINED_CKPT = "/home/cchen/桌面/diff2flow-smalldata (副本)/logs/celeba_identity_v1_2026-01-12-19-17-48/checkpoints/last.ckpt"
    CONFIG_PATH = "/home/cchen/桌面/diff2flow-smalldata (副本)/logs/celeba_identity_v1_2026-01-12-19-17-48/config.yaml"

    # 資料路徑
    TRAIN_DATA = "/home/cchen/桌面/MM-CelebA-HQ-Dataset-main\split/train"
    VAL_DATA = "/home/cchen/桌面/MM-CelebA-HQ-Dataset-main\split/val"

    # 屬性檔案（設成 None 自動找）
    TRAIN_ATTRIBUTES = None
    VAL_ATTRIBUTES = None

    OUTPUT_DIR = "/home/cchen/桌面/diff2flow-smalldata (副本)/editing_finetuned"

    # ==========================================
    # 訓練參數（重要！）
    # ==========================================
    MAX_EPOCHS = 30          # 訓練輪數
    BATCH_SIZE = 4           # 批次大小
    LEARNING_RATE = 2e-5     # 學習率（提高了！）
    NOISE_LEVEL = 0.3        # 雜訊強度
    DROP_RATIO = 0.5         # Prompt 丟棄率（提高了！）

    # 訓練模式
    USE_STANDARD_SPLIT = True
    USE_COMBINED = False

    if USE_STANDARD_SPLIT:
        print("\n" + "="*60)
        print("使用標準分割模式")
        print(f"  訓練: {TRAIN_DATA}")
        print(f"  驗證: {VAL_DATA}")
        print("="*60 + "\n")

        train_editing_finetuner(
            pretrained_ckpt=PRETRAINED_CKPT,
            config_path=CONFIG_PATH,
            train_data_root=TRAIN_DATA,
            val_data_root=VAL_DATA,
            train_attributes_file=TRAIN_ATTRIBUTES,
            val_attributes_file=VAL_ATTRIBUTES,
            output_dir=OUTPUT_DIR,
            use_combined_data=False,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            noise_level=NOISE_LEVEL,
            drop_ratio=DROP_RATIO
        )

    elif USE_COMBINED:
        print("\n" + "="*60)
        print("使用合併模式（train + val）")
        print(f"  訓練: {TRAIN_DATA} + {VAL_DATA}")
        print("="*60 + "\n")

        train_editing_finetuner(
            pretrained_ckpt=PRETRAINED_CKPT,
            config_path=CONFIG_PATH,
            train_data_root=TRAIN_DATA,
            val_data_root=VAL_DATA,
            train_attributes_file=TRAIN_ATTRIBUTES,
            val_attributes_file=VAL_ATTRIBUTES,
            output_dir=OUTPUT_DIR,
            use_combined_data=True,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            noise_level=NOISE_LEVEL,
            drop_ratio=DROP_RATIO
        )

    print("\n" + "="*60)
    print("微調完成！")
    print(f"Checkpoints: {OUTPUT_DIR}/checkpoints/")
    print("="*60)