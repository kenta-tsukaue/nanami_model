import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, input_ids, attention_masks, labels = [], [], [], []

    for image, (input_id, attention_mask), label in batch:
        images.append(image)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)
    
    # 画像をテンソルに変換
    images = torch.stack(images).float()  # 明示的にfloat32に変換
    
    # input_idsとattention_maskをパディングしてバッチ化
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    # ラベルをテンソルに変換（回帰タスク用にfloat32）
    labels = torch.stack(labels).float()

    return images, {'input_ids': input_ids, 'attention_mask': attention_masks}, labels

class FontDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, tokenizer=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 画像タイトルと画像パスを取得
        title = self.data.iloc[idx]['title']
        number = self.data.iloc[idx]['number']
        img_path = f"{self.img_dir}/img_data_{title}/IMG_{number}.jpg"
        
        # 画像の読み込みと前処理
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # タイトルのトークン化（BERT用）
        text = self.data.iloc[idx]['image_title']
        text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # BERTのトークンをフラットに変換
        text_input_ids = text_tokens['input_ids'].squeeze(0)
        text_attention_mask = text_tokens['attention_mask'].squeeze(0)
        
        # ラベルを取得（回帰タスク用に連続値として読み込む）
        labels = self.data.iloc[idx][['DelaGothic', 'HachiMaruPop', 'KaiseiDecol', 'NotoSansJP', 
                                      'NotoSerifJP', 'Reggae', 'Stick']].values.astype(float)
        
        return image, (text_input_ids, text_attention_mask), torch.tensor(labels).float()

def create_dataloaders(csv_file, img_dir, tokenizer, batch_size=32, test_split=0.2):
    # 訓練データ用の前処理（データ拡張を含む）
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.7, 1.0), ratio=(1.0, 1.0), interpolation=3),  # クリッピングとリサイズ
        transforms.RandomChoice([
            transforms.RandomRotation(angle) for angle in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # テストデータ用の前処理（データ拡張なし）
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # データセットの作成
    full_dataset = FontDataset(csv_file, img_dir, transform=None, tokenizer=tokenizer)
    
    # データの80/20分割
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 訓練データとテストデータにそれぞれのtransformを適用
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader