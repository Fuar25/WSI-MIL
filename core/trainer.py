import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold
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
            for batch in pbar:
                # Support datasets returning (features, label, filename, patient_id)
                if len(batch) == 4:
                    features, label, _, _ = batch
                else:
                    features, label, _ = batch
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
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    features, label, sample_ids, patient_ids = batch
                else:
                    features, label, sample_ids = batch
                    patient_ids = ['unknown'] * len(sample_ids)
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
                all_patient_ids.extend(patient_ids)
                    
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
            return avg_loss, accuracy, auc, {
                'sample_ids': all_sample_ids,
                'patient_ids': all_patient_ids,
                'probs': all_probs,
                'labels': all_labels
            }
        return avg_loss, accuracy, auc

    def _aggregate_patient_results(self, results_df: pd.DataFrame):
        """Aggregate slide-level predictions to patient-level using configured strategy."""
        strategy = self.training_cfg.voting_strategy

        # Soft voting: mean probabilities per patient
        if strategy == 'average':
            if self.model_cfg.num_classes == 1:
                grouped = results_df.groupby('patient_id')['prob_positive'].mean().reset_index()
                grouped['prediction'] = (grouped['prob_positive'] > 0.5).astype(int)
                # For binary AUC, need prob_positive as score
                return grouped
            else:
                prob_cols = [c for c in results_df.columns if c.startswith('prob_class_')]
                grouped = results_df.groupby('patient_id')[prob_cols].mean().reset_index()
                grouped['prediction'] = grouped[prob_cols].values.argmax(axis=1)
                return grouped

        # Hard voting: majority on predicted class; tie-break by mean prob if available
        elif strategy == 'majority':
            if self.model_cfg.num_classes == 1:
                voted = results_df.groupby('patient_id')['prediction'].agg(lambda x: int(x.sum() >= (len(x) / 2))).reset_index()
                # Add averaged prob for AUC stability
                voted = voted.merge(results_df.groupby('patient_id')['prob_positive'].mean().reset_index(), on='patient_id', how='left')
                return voted
            else:
                def vote_row(group):
                    counts = group['prediction'].value_counts()
                    top_classes = counts[counts == counts.max()].index.tolist()
                    if len(top_classes) == 1:
                        return int(top_classes[0])
                    # tie-breaker by summed probs if available
                    prob_cols = [c for c in results_df.columns if c.startswith('prob_class_')]
                    if prob_cols:
                        probs_sum = group[prob_cols].sum()
                        return int(probs_sum.idxmax().replace('prob_class_', ''))
                    return int(top_classes[0])

                voted = results_df.groupby('patient_id').apply(vote_row).reset_index(name='prediction')
                # If probability columns exist, keep their mean for possible metrics
                prob_cols = [c for c in results_df.columns if c.startswith('prob_class_')]
                if prob_cols:
                    voted = voted.merge(results_df.groupby('patient_id')[prob_cols].mean().reset_index(), on='patient_id', how='left')
                return voted

        else:
            raise ValueError(f"Unknown voting strategy: {strategy}")


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
        """Group-aware K-Fold CV to avoid patient leakage and report patient-level metrics."""
        fold_results = []
        os.makedirs(self.logging_cfg.save_dir, exist_ok=True)

        print(f"Starting {k_folds}-Fold Cross-Validation (grouped by patient)...")
        print(f"Each fold: Test set = 1/{k_folds}, Train:Val ≈ 9:1 (patient-aware)")
        cv_start_time = time.time()

        all_indices = np.arange(len(self.dataset))
        patient_ids = np.array(self.dataset.get_patient_ids())
        labels = np.array([item['label'] for item in self.dataset.data])

        # Prefer stratified grouping for binary; fallback to group-only
        if self.model_cfg.num_classes == 1:
            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=self.training_cfg.seed)
            split_iter = splitter.split(all_indices, labels, groups=patient_ids)
        else:
            splitter = GroupKFold(n_splits=k_folds)
            split_iter = splitter.split(all_indices, groups=patient_ids)

        all_fold_results_dfs = []

        for fold, (train_val_ids, test_ids) in enumerate(split_iter):
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1} Start!")
            print(f"{'='*70}")

            # Patient-aware split for validation inside train_val
            rng = np.random.default_rng(self.training_cfg.seed + fold)
            train_val_patients = patient_ids[train_val_ids]
            unique_patients = np.unique(train_val_patients)
            rng.shuffle(unique_patients)
            val_patient_count = max(1, int(len(unique_patients) * self.training_cfg.val_ratio))
            val_patients = set(unique_patients[:val_patient_count])

            val_mask = np.array([pid in val_patients for pid in train_val_patients])
            val_ids = train_val_ids[val_mask]
            train_ids = train_val_ids[~val_mask]

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
            patient_acc = float('nan')
            patient_auc = float('nan')

            if self.logging_cfg.log_test_results:
                results_df = pd.DataFrame({
                    'filename': test_details['sample_ids'],
                    'patient_id': test_details['patient_ids'],
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

                # Patient-level aggregation
                patient_df = self._aggregate_patient_results(results_df)
                # Ground-truth per patient via majority label (should be consistent in clean data)
                patient_labels = results_df.groupby('patient_id')['label'].agg(lambda x: np.bincount(x.astype(int)).argmax()).reset_index()
                patient_df = patient_df.merge(patient_labels, on='patient_id', how='left', suffixes=('', '_gt'))
                patient_df = patient_df.rename(columns={'label': 'patient_label'})

                # Patient-level metrics
                try:
                    if self.model_cfg.num_classes == 1:
                        patient_auc = roc_auc_score(patient_df['patient_label'].astype(int), patient_df['prob_positive'])
                    else:
                        prob_cols = [c for c in patient_df.columns if c.startswith('prob_class_')]
                        patient_auc = roc_auc_score(patient_df['patient_label'].astype(int), patient_df[prob_cols], multi_class='ovr', average='macro')
                except Exception as e:
                    print(f"Warning: Patient-level AUC failed to compute: {e}")
                    patient_auc = 0.0

                patient_acc = (patient_df['prediction'].astype(int) == patient_df['patient_label'].astype(int)).mean()

                csv_path = os.path.join(self.logging_cfg.save_dir, f'fold_{fold+1}_{self.logging_cfg.test_results_csv}')
                results_df.to_csv(csv_path, index=False)
                patient_csv_path = os.path.join(self.logging_cfg.save_dir, f'fold_{fold+1}_patient_{self.logging_cfg.test_results_csv}')
                patient_df.to_csv(patient_csv_path, index=False)
                print(f"  Test results saved to {csv_path}")
                print(f"  Patient-level results saved to {patient_csv_path}")

                all_fold_results_dfs.append(results_df)

            fold_results.append({
                'fold': fold + 1,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'patient_acc': patient_acc if self.logging_cfg.log_test_results else None,
                'patient_auc': patient_auc if self.logging_cfg.log_test_results else None
            })

            print(f"  Fold {fold+1} Results: Best Epoch={best_epoch}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Test AUC={test_auc:.4f}, Patient Acc={patient_acc:.4f}, Patient AUC={patient_auc:.4f}")

        total_time = time.time() - cv_start_time

        if self.logging_cfg.log_test_results and all_fold_results_dfs:
            full_df = pd.concat(all_fold_results_dfs, ignore_index=True)
            full_csv_path = os.path.join(self.logging_cfg.save_dir, f'full_{self.logging_cfg.test_results_csv}')
            full_df.to_csv(full_csv_path, index=False)
            print(f"  Full test results saved to {full_csv_path}")

        # Aggregate results
        test_aucs = [r['test_auc'] for r in fold_results]
        test_accs = [r['test_acc'] for r in fold_results]
        patient_aucs = [r['patient_auc'] for r in fold_results if np.isfinite(r['patient_auc'])]
        patient_accs = [r['patient_acc'] for r in fold_results if np.isfinite(r['patient_acc'])]
        best_epochs = [r['best_epoch'] for r in fold_results]

        print(f"\n{'='*70}")
        print(f"Cross-Validation Results:")
        print(f"{'='*70}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Best Epoch={r['best_epoch']:3d}, "
                  f"Test AUC={r['test_auc']:.4f}, Test Acc={r['test_acc']:.4f}, "
                  f"Patient AUC={r['patient_auc']:.4f}, Patient Acc={r['patient_acc']:.4f}")
        print(f"{'-'*70}")
        print(f"  Average Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"  Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        if patient_aucs:
            print(f"  Average Patient AUC: {np.mean(patient_aucs):.4f} ± {np.std(patient_aucs):.4f}")
            print(f"  Average Patient Acc: {np.mean(patient_accs):.4f} ± {np.std(patient_accs):.4f}")
        print(f"  Average Best Epoch: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
        print(f"  Total Time: {total_time/60:.2f} minutes")
        print(f"{'='*70}")

        return fold_results

