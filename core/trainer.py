import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Create convenient access to nested configs
        self.model_cfg = config.model
        self.training_cfg = config.training
        self.logging_cfg = config.logging
        
        self.device = torch.device(self.training_cfg.device)
        
    def _train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(loader, desc="  Training one Epoch", unit="batch", leave=False) as pbar:
            for features, label, _ in pbar:
                features = features.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                
                # Reshape label for model if binary classification to match logits [B, 1]
                model_label = label.view(-1, 1) if self.model_cfg.num_classes == 1 else label
                
                # Pass label and criterion to model to allow internal loss calculation (e.g. CLAM instance loss)
                logits, loss, _ = model(features, label=model_label, loss_fn=criterion)

                if loss is None:
                    # Fallback to external loss calculation if model didn't return one
                    if self.model_cfg.num_classes == 1:
                        loss = criterion(logits.view(-1), label)
                    else:
                        loss = criterion(logits, label.long())

                if self.model_cfg.num_classes == 1:
                    preds = (torch.sigmoid(logits) > 0.5).float().view(-1)
                else:
                    preds = torch.argmax(logits, dim=1)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)
                correct += (preds == label).sum().item()
                total += features.size(0)
        
        return total_loss / total, correct / total

    def _evaluate_with_auc(self, model, loader, criterion, return_details=False):
        """Evaluate model and compute AUC for binary/multi-class classification"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        all_sample_ids = []
        
        with torch.no_grad():
            for features, label, sample_ids in loader:
                features = features.to(self.device)
                label = label.to(self.device)
                
                # Reshape label for model if binary classification to match logits [B, 1]
                model_label = label.view(-1, 1) if self.model_cfg.num_classes == 1 else label
                
                # Pass label and criterion to model (though loss is not used for backprop here, it might be logged)
                logits, loss, _ = model(features, label=model_label, loss_fn=criterion)
                
                if loss is None:
                    if self.model_cfg.num_classes == 1:
                        loss = criterion(logits.view(-1), label)
                    else:
                        loss = criterion(logits, label.long())
                
                if self.model_cfg.num_classes == 1:
                    probs = torch.sigmoid(logits).view(-1)
                    preds = (probs > 0.5).float()
                    
                    # Store labels and probabilities
                    all_labels.extend(label.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    loss = criterion(logits, label.long())
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    # Store labels and probabilities (keep multi-class probs as matrix)
                    all_labels.extend(label.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                
                all_sample_ids.extend(sample_ids)
                    
                total_loss += loss.item() * features.size(0)
                correct += (preds == label).sum().item()
                total += features.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Compute AUC
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        try:
            if self.model_cfg.num_classes == 1:
                # Binary classification: labels should be 0/1, probs is a 1D array
                all_labels_int = all_labels.astype(int)
                auc = roc_auc_score(all_labels_int, all_probs)
            else:
                # Multi-class AUC (one-vs-rest)
                # all_probs shape: (n_samples, n_classes)
                # all_labels: (n_samples,) with integer class labels
                all_labels_int = all_labels.astype(int)
                auc = roc_auc_score(all_labels_int, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"\nWarning: Could not compute AUC. Error: {e}")
            print(f"Labels shape: {all_labels.shape}, unique values: {np.unique(all_labels)}")
            print(f"Probs shape: {all_probs.shape}, range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
            auc = 0.0
            
        if return_details:
            return avg_loss, accuracy, auc, {'sample_ids': all_sample_ids, 'probs': all_probs, 'labels': all_labels}
        return avg_loss, accuracy, auc


    def train_full_dataset(self):
        """
        Train on full dataset for deployment model (no validation).
        """
        print(f"Training on full dataset ({len(self.dataset)} samples) for deployment...")
        
        train_loader = DataLoader(self.dataset, batch_size=self.training_cfg.batch_size, shuffle=True)
        
        model = self.model(
            input_dim=self.model_cfg.input_dim,
            hidden_dim=self.model_cfg.hidden_dim,
            num_classes=self.model_cfg.num_classes,
            n_heads=self.model_cfg.n_heads,
            dropout=self.model_cfg.dropout,
            gated=self.model_cfg.gated
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.training_cfg.learning_rate, weight_decay=self.training_cfg.weight_decay)
        
        if self.model_cfg.num_classes == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
        
        epoch_start_time = time.time()
        epoch_pbar = tqdm(range(self.training_cfg.best_epochs), desc="Epochs", unit="epoch", leave=False)
        for epoch in epoch_pbar:
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)

            epoch_time = time.time() - epoch_start_time
            avg_time_per_epoch = epoch_time / (epoch + 1)
            remaining_epochs = self.training_cfg.epochs - epoch - 1
            eta_seconds = avg_time_per_epoch * remaining_epochs
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)

            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "ETA": f"{eta_min}:{eta_sec:02d}"
            })
        
        # Save final model
        model_path = os.path.join(self.logging_cfg.save_dir, 'model_deployment.pth')
        torch.save(model.state_dict(), model_path)
        
        total_time = time.time() - epoch_start_time
        print(f"\nTraining complete in {total_time/60:.2f} minutes. Deployment model saved to {model_path}")

    def cross_validate(self, k_folds=5):
        """
        Perform K-Fold Cross-Validation with proper train/val/test split.
        
        Logic:
        - For each fold, one part is held out as test set
        - Remaining data is split into train (90%) and validation (10%)
        - Train for N epochs, monitor validation performance
        - Early stop based on validation loss
        - Evaluate best checkpoint on test set
        - Compute AUC for each fold on test set
        """
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.training_cfg.seed)
        fold_results = []
        os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
        
        print(f"Starting {k_folds}-Fold Cross-Validation...")
        print(f"Each fold: Test set = 1/{k_folds}, Train:Val = 9:1")
        cv_start_time = time.time()
        
        all_indices = np.arange(len(self.dataset))
        fold_generator = list(enumerate(kfold.split(all_indices)))
        
        all_fold_results_dfs = []
        
        for fold, (train_val_ids, test_ids) in fold_generator:
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1} Start!")
            print(f"{'='*70}")
            
            # Further split train_val into train and validation
            train_val_size = len(train_val_ids)
            val_size = int(train_val_size * self.training_cfg.val_ratio)
            train_size = train_val_size - val_size
            
            # Shuffle and split
            np.random.seed(self.training_cfg.seed + fold)
            np.random.shuffle(train_val_ids)
            train_ids = train_val_ids[:train_size]
            val_ids = train_val_ids[train_size:]
            
            train_subset = Subset(self.dataset, train_ids)
            val_subset = Subset(self.dataset, val_ids)
            test_subset = Subset(self.dataset, test_ids)
            
            train_loader = DataLoader(train_subset, batch_size=self.training_cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.training_cfg.batch_size, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=self.training_cfg.batch_size, shuffle=False)
            
            print(f"  Fold {fold+1}: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}\n")
            
            # Initialize model
            model = self.model(
                input_dim=self.model_cfg.input_dim,
                hidden_dim=self.model_cfg.hidden_dim,
                num_classes=self.model_cfg.num_classes,
                n_heads=self.model_cfg.n_heads,
                dropout=self.model_cfg.dropout,
                gated=self.model_cfg.gated
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=self.training_cfg.learning_rate, weight_decay=self.training_cfg.weight_decay)
            
            if self.model_cfg.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
                
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            fold_start_time = time.time()
            
            # Training loop with early stopping
            with tqdm(range(self.training_cfg.epochs), desc=f"  Epochs", leave=False, unit="epoch") as epoch_pbar:
                for epoch in epoch_pbar:
                    train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
                    val_loss, val_acc, val_auc = self._evaluate_with_auc(model, val_loader, criterion)
                    
                    epoch_time = time.time() - fold_start_time
                    avg_time_per_epoch = epoch_time / (epoch + 1)
                    remaining_epochs = self.training_cfg.epochs - epoch - 1
                    eta_seconds = avg_time_per_epoch * remaining_epochs
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    
                    epoch_pbar.set_postfix({
                        "train_loss": f"{train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "val_auc": f"{val_auc:.4f}",
                        "ETA": f"{eta_min}:{eta_sec:02d}"
                    })
                    
                    # Early stopping based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        patience_counter = 0
                        # Save best checkpoint
                        best_checkpoint_path = os.path.join(self.logging_cfg.save_dir, f'model_fold_{fold+1}_best.pth')
                        torch.save(model.state_dict(), best_checkpoint_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= self.training_cfg.patience:
                            tqdm.write(f"\n  Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
                            break
            
            # Load best checkpoint and evaluate on test set
            model.load_state_dict(torch.load(best_checkpoint_path))
            test_loss, test_acc, test_auc, test_details = self._evaluate_with_auc(model, test_loader, criterion, return_details=True)
            
            if self.logging_cfg.log_test_results:
                results_df = pd.DataFrame({
                    'filename': test_details['sample_ids'],
                    'label': test_details['labels'],
                })
                
                if self.model_cfg.num_classes == 1:
                    probs = test_details['probs']
                    results_df['prob_negative'] = 1 - probs
                    results_df['prob_positive'] = probs
                    results_df['prediction'] = (probs > 0.5).astype(int)
                else:
                    probs = test_details['probs']
                    for c in range(self.model_cfg.num_classes):
                        results_df[f'prob_class_{c}'] = probs[:, c]
                    results_df['prediction'] = np.argmax(probs, axis=1)
                        
                csv_path = os.path.join(self.logging_cfg.save_dir, f'fold_{fold+1}_{self.logging_cfg.test_results_csv}')
                results_df.to_csv(csv_path, index=False)
                print(f"  Test results saved to {csv_path}")
                
                all_fold_results_dfs.append(results_df)
            
            fold_results.append({
                'fold': fold + 1,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc
            })
            
            print(f"  Fold {fold+1} Results: Best Epoch={best_epoch}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Test AUC={test_auc:.4f}")
            
        total_time = time.time() - cv_start_time
        
        if self.logging_cfg.log_test_results and all_fold_results_dfs:
            full_df = pd.concat(all_fold_results_dfs, ignore_index=True)
            full_csv_path = os.path.join(self.logging_cfg.save_dir, f'full_{self.logging_cfg.test_results_csv}')
            full_df.to_csv(full_csv_path, index=False)
            print(f"  Full test results saved to {full_csv_path}")
        
        # Aggregate results
        test_aucs = [r['test_auc'] for r in fold_results]
        test_accs = [r['test_acc'] for r in fold_results]
        best_epochs = [r['best_epoch'] for r in fold_results]
        
        print(f"\n{'='*70}")
        print(f"Cross-Validation Results:")
        print(f"{'='*70}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Best Epoch={r['best_epoch']:3d}, "
                  f"Test AUC={r['test_auc']:.4f}, Test Acc={r['test_acc']:.4f}")
        print(f"{'-'*70}")
        print(f"  Average Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"  Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"  Average Best Epoch: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
        print(f"  Total Time: {total_time/60:.2f} minutes")
        print(f"{'='*70}")
        
        return fold_results

