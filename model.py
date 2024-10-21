import torch
import torch.nn as nn
from transformers import BertModel

class MultiModalClassifier(nn.Module):
    def __init__(self):
        super(MultiModalClassifier, self).__init__()
        # 画像処理用のCNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # テキスト処理用のBERT (事前学習済みのBERTモデルを利用)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # BERTのパラメータをフリーズ（固定）
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 結合後の全結合層
        self.fc1 = nn.Linear(2 * 64 * 64 + 768, 128)  # CNN出力 + BERT出力
        self.fc2 = nn.Linear(128, 7)  # 7つのフォントカテゴリ

    def forward(self, image, text_tokens):
        # CNNで画像処理
        x_image = torch.relu(self.conv1(image))
        #print(f"After conv1: {x_image.shape}")  # conv1の出力形状を確認
        x_image = self.pool(x_image)
        x_image = torch.relu(self.conv2(x_image))
        #print(f"After conv2: {x_image.shape}")  # conv2の出力形状を確認
        x_image = self.pool(x_image)
        
        # BERTでテキスト処理 (事前学習済みの特徴を抽出)
        with torch.no_grad():  # BERT部分は勾配を計算しない
            bert_output = self.bert(**text_tokens).pooler_output  # [CLS]トークンの出力
        
        # 画像とテキストの特徴を結合する前に形状を確認
        #print(f"x_image shape: {x_image.shape}")
        #print(f"bert_output shape: {bert_output.shape}")
        
        # 画像とテキストの特徴を結合
        x_image = x_image.view(x_image.size(0), -1)  # viewでバッチサイズに合わせる
        #print(f"x_image shape: {x_image.shape}")
        x = torch.cat((x_image, bert_output), dim=1)

        #print(f"x shape: {x.shape}")
        
        # 結合後に全結合層を通す
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x