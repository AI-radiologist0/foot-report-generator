import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

# -----------------------------
# 설정
# -----------------------------
data_dir = "data/foot/pair/pair"
batch_size = 16
num_epochs = 10
lr = 1e-4
early_stop_patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# -----------------------------
# 데이터 전처리 및 분할
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 전체 데이터셋 로드
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
total_size = len(full_dataset)
# 학습/검증 셋 분할 (80/20)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 모델 정의
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -----------------------------
# 손실함수 및 옵티마이저
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

# -----------------------------
# 학습 루프 with Early Stopping
# -----------------------------
best_val_acc = 0.0
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Early Stopping 체크
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnet18_best_include2.pth")
        print("✅ 모델 저장됨 (최고 검증 정확도 갱신)")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⏸️  Early stopping 카운터 증가: {patience_counter}/{early_stop_patience}")
        if patience_counter >= early_stop_patience:
            print("🛑 Early stopping triggered")
            break

#-----------------------
# Test
#------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"[test] {test_acc:.4f}")


# -----------------------------
# 모델 저장
# -----------------------------
torch.save(model.state_dict(), "resnet18_pair_classifier_include2.pth")
