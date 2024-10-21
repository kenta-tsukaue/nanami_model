import torch
import torch.nn as nn
import torch.optim as optim
from model import MultiModalClassifier
from transformers import BertTokenizer
from dataloader import create_dataloaders

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np  # 評価関数で使用

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, text_tokens, labels in dataloader:
            images = images.to(device)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            labels = labels.to(device).float()  # 明示的にfloat32に変換
            
            outputs = model(images, text_tokens).float()  # 明示的にfloat32に変換
            
            # 出力をCPUに移動し、リストに追加
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # すべてのバッチの予測とラベルを結合
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 評価指標の計算
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    
    return mse, mae

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MultiModalClassifier().to(device)
    criterion = nn.MSELoss()  # 回帰タスク用の損失関数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    csv_file = "data/data.csv"
    img_dir = "data/imgs"
    batch_size = 32

    # データローダの作成
    train_loader, test_loader = create_dataloaders(csv_file, img_dir, tokenizer, batch_size)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, text_tokens, labels in train_loader:
            images = images.to(device).float()  # 明示的にfloat32に変換
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            labels = labels.to(device).float()  # 明示的にfloat32に変換
            
            optimizer.zero_grad()

            # モデルの出力を計算
            outputs = model(images, text_tokens).float()  # 明示的にfloat32に変換
            
            # ロスを計算して逆伝播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 訓練データでの評価
    train_mse, train_mae = evaluate(model, train_loader, device)
    print(f"Training MSE: {train_mse:.4f}, Training MAE: {train_mae:.4f}")

    # テストデータでの評価
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

    # モデルの保存（オプション）
    torch.save(model.state_dict(), 'model_regression.pth')

if __name__ == "__main__":
    run()