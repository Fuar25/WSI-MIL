from WSIFeatureDataset import NPYDataset, NPYDataset_with_dirname
from typing import Optional, Literal, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import  DataLoader, Subset
from tqdm import tqdm
from torch import nn
from SlideEncoder import ABMIL

def train_slide_encoder(model:nn.Module,
                        dataset: Union[NPYDataset, NPYDataset_with_dirname],
                        train_indices,
                        epochs:int,
                        device:str = 'cuda:1',
                        train_val_ratio = 0.9,
                        patience = 5):

    model = model.to(device).train()
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    train_label = np.array(dataset.label_list)[train_indices]

    train_indices, val_indices = train_test_split(
        train_indices,
        train_size=train_val_ratio,
        random_state=42,
        shuffle=True,
        stratify=train_label
    )

    train_dataset = Subset(dataset, indices=train_indices)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    best_auc = 0.0
    best_model = model.state_dict()
    stop_counters = 0

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for data, label, _ in pbar:
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.close()
        val_auc = test_slide_encoder(model, dataset, test_indices=val_indices, device=device)
        tqdm.write(f"EPOCH{epoch+1} finished, loss={total_loss / len(train_loader):.4f}, AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_model = model.state_dict()
            best_auc = val_auc
            stop_counters = 0
        else:
            stop_counters += 1
        if stop_counters >= patience:
            tqdm.write(f"Up to the patience, stop! Best AUC = {best_auc}")
            break

    return best_model


def test_slide_encoder(model: nn.Module,
                       dataset: Union[NPYDataset, NPYDataset_with_dirname],
                       test_indices,
                       device: str):

    model = model.to(device).eval()
    criterion = BCEWithLogitsLoss()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    test_dataset = Subset(dataset, indices=test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f'Evaluating')
        for data, label, _ in pbar:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            probs = nn.Sigmoid()(logits)
            loss = criterion(logits, label)
            total_loss += loss.item()

            all_labels.extend(label.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().detach().numpy().flatten())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.close()
    auc = roc_auc_score(all_labels, all_probs)
    return auc

def T_SNE_visualization(model:nn.Module, dataset: Union[NPYDataset, NPYDataset_with_dirname], title: Optional[str] = None, device:str = 'cuda:1'):
    all_slide_emb = []
    all_labels = []
    all_scores = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data, label, score in dataloader:
            slide_emb = model.slide_encoder(data, device=device)
            all_slide_emb.append(slide_emb.cpu().numpy())
            all_labels.append(int(label.cpu().numpy().item()))
            all_scores.append(float(score.cpu().numpy().item()))
        all_slide_emb_array = np.vstack(all_slide_emb)
        all_slide_emb_2D = TSNE(n_components=2, learning_rate='auto').fit_transform(all_slide_emb_array)
        fig, ax = plt.subplots(figsize=(16, 8))

        df = pd.DataFrame({
            'x':all_slide_emb_2D[:, 0],
            'y':all_slide_emb_2D[:, 1],
            'label': all_labels,
            'score': all_scores
        })

        sns.scatterplot(data=df, x='x', y='y', hue='score', palette='turbo', ax=ax)
        if title is not None:
            ax.set_title(title)
        plt.show()

def plot(model:nn.Module, dataset: Union[NPYDataset, NPYDataset_with_dirname], indices, title: Optional[str] = None, device:str = 'cuda:1'):
    all_labels = []
    all_preds = []
    all_probs = []

    subset = Subset(dataset, indices=indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    model = model.to(device)

    with torch.no_grad():
        for data, label, score in dataloader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            probs = nn.Sigmoid()(logits)
            all_probs.append(probs.cpu().numpy().item())
            all_labels.append(int(label.cpu().numpy().item()))
            preds = (probs.item()) > 0.5
            all_preds.append(preds)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar_kws={"shrink": .8})
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()


    from sklearn.metrics import roc_curve, auc, RocCurveDisplay
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fpr_smooth = np.linspace(0, 1, 1000)
    tpr_smooth = np.interp(fpr_smooth, fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_smooth, tpr_smooth, color='darkblue', lw=3, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')  # 对角线（随机猜测）

    # 美化
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def main():
    root = '/home/william/Desktop/gml/GML/features_wtih_raw_coords'
    # root = '/mnt/gml/PD-L1/previous/features/cpath_feature/01'
    csv_path = '/home/william/Desktop/gml/ALL_with_score.csv'
    # npydataset = NPYDataset(root=root, csv_path=csv_path, score=True)
    npydataset = NPYDataset_with_dirname(root=root, csv_path=csv_path, score=True)
    indices = list(range(len(npydataset)))
    labels = npydataset.label_list
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf_AUC = []

    for kfold, (train_indices, test_indices) in enumerate(kf.split(indices, labels)):
        model = ABMIL(feature_dim=768, atte_emb_dim=256, hidden_dim=512, class_num=2)
        print(f"=============={kfold} fold begins!===============")
        best_model_state = train_slide_encoder(model, npydataset, device=device, epochs=100, train_indices=train_indices, patience=10)
        model.load_state_dict(best_model_state)
        test_auc = test_slide_encoder(model, npydataset, device=device, test_indices=test_indices)
        kf_AUC.append(test_auc)
        print(f"=============={kfold} fold ends!=================")

        # T_SNE_visualization(model, train_loader, title="Train", device=device)
        # T_SNE_visualization(model, val_loader, title="Test", device=device)

        plot(model, npydataset, indices=test_indices, device=device)

    average_auc = sum(kf_AUC) / len(kf_AUC)
    print("\n")
    print(f"KFold Average AUC = {average_auc}")

if __name__ == '__main__':
    main()
