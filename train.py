import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Alt kümelere bağımsız transform uygulamak için yardımcı sınıf
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def main():
    # 1. Donanım ve Hiperparametreler (Rapor Tablosu İçin)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 8
    IMG_SIZE = 224

    print(f"Kullanılan Donanım: {DEVICE}")

    # 2. Veri Dönüşümleri (Sağ ve Sol El Uyumlu Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),  # Hem sağ hem sol eli tanıması için
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Veri Setini Yükleme ve %70 - %15 - %15 Bölme
    base_dataset = datasets.ImageFolder(root="./dataset")
    class_names = base_dataset.classes
    num_classes = len(class_names)

    total_size = len(base_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    raw_train, raw_val, raw_test = random_split(
        base_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_data = SubsetWithTransform(raw_train, transform=train_transform)
    val_data = SubsetWithTransform(raw_val, transform=eval_transform)
    test_data = SubsetWithTransform(raw_test, transform=eval_transform)

    # Windows uyumluluğu için num_workers=0 kullanıyoruz
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Toplam Sınıf: {num_classes} | Train: {train_size}, Val: {val_size}, Test: {test_size}\n")

    # 4. Model Mimarisi: ResNet18 (Transfer Learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Eğitim Döngüsü ve Süre Ölçümü
    best_val_acc = 0.0
    start_train_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Doğrulama (Validation)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / val_size
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names
            }, "best_asl_model.pth")

    total_train_time = time.time() - start_train_time
    print(f"\nToplam Eğitim Süresi (Training Time): {total_train_time:.2f} saniye")

    # 6. Test Kümesi Değerlendirmesi ve Inference Time Ölçümü
    checkpoint = torch.load("best_asl_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()

            # Tek görüntü başına ms cinsinden tahmin süresi
            inference_times.append(((t1 - t0) / images.size(0)) * 1000)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    avg_inference_ms = np.mean(inference_times)

    print("\n================ ÖDEV RAPORU TEST SONUÇLARI ================")
    print(f"Test Accuracy  : {acc:.4f} (%{acc*100:.2f})")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1-Score : {f1:.4f}")
    print(f"Inference Time : {avg_inference_ms:.2f} ms / görüntü")
    print("=============================================================")

    # Rapor için Confusion Matrix (Karmaşıklık Matrisi) Çizdirme
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
    plt.title("ASL Alphabet Confusion Matrix")
    plt.xlabel("Tahmin Edilen Harf")
    plt.ylabel("Gerçek Harf")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Karmaşıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()