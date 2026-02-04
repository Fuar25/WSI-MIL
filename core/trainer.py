import os
import time
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import core.voting_strategies  # noqa: F401
from config.defaults import BINARY_THRESHOLD
from core.registry import create_voting_strategy

class Trainer:
    def __init__(self, model_builder: Callable[[], nn.Module], dataset, config):
        self.model_builder = model_builder
        self.dataset = dataset
        self.config = config

        self.model_cfg = config.model
        self.training_cfg = config.training
        self.logging_cfg = config.logging

        self.device = torch.device(self.training_cfg.device)
        self.voting_strategy = self._init_voting_strategy()

    def _init_voting_strategy(self):
        try:
            return create_voting_strategy(
                self.training_cfg.voting_strategy,
                **(self.training_cfg.voting_config or {}),
            )
        except KeyError as exc:
            print(f"Warning: {exc}. Falling back to average voting.")
            return create_voting_strategy('average', threshold=BINARY_THRESHOLD)

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    @staticmethod
    def _model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key not in {'label', 'sample_id', 'patient_id'}
        }

    def _train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(loader, desc="  Training one Epoch", unit="batch", leave=False) as pbar:
            for raw_batch in pbar:
                batch = self._move_to_device(raw_batch)
                inputs = self._model_inputs(batch)
                labels = batch['label']

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs['logits']

                if self.model_cfg.num_classes == 1:
                    logits = logits.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(logits, labels)
                    probs = torch.sigmoid(logits)
                    preds = (probs > BINARY_THRESHOLD).float()
                else:
                    loss = criterion(logits, labels.long())
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (preds == labels).sum().item()
                total += batch_size

        return total_loss / total, correct / total

    def _evaluate_with_auc(self, model, loader, criterion, return_details=False):
        """Evaluate model and compute AUC for binary/multi-class classification."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        all_sample_ids = []
        all_patient_ids = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._move_to_device(raw_batch)
                inputs = self._model_inputs(batch)
                labels = batch['label']
                sample_ids = batch.get('sample_id', ['unknown'] * len(labels))
                patient_ids = batch.get('patient_id', sample_ids)

                outputs = model(inputs)
                logits = outputs['logits']

                if self.model_cfg.num_classes == 1:
                    logits = logits.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(logits, labels)
                    probs = torch.sigmoid(logits)
                    preds = (probs > BINARY_THRESHOLD).float()
                    all_probs.extend(probs.cpu().numpy())
                else:
                    loss = criterion(logits, labels.long())
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.extend(probs.cpu().numpy())

                all_labels.extend(labels.cpu().numpy())
                all_sample_ids.extend(sample_ids)
                all_patient_ids.extend(patient_ids)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (preds == labels).sum().item()
                total += batch_size

        avg_loss = total_loss / total
        accuracy = correct / total

        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        try:
            if self.model_cfg.num_classes == 1:
                auc = roc_auc_score(all_labels_np.astype(int), all_probs_np)
            else:
                auc = roc_auc_score(
                    all_labels_np.astype(int),
                    all_probs_np,
                    multi_class='ovr',
                    average='macro',
                )
        except Exception as e:
            print(f"\nWarning: Could not compute AUC. Error: {e}")
            print(f"Labels shape: {all_labels_np.shape}, unique values: {np.unique(all_labels_np)}")
            print(f"Probs shape: {all_probs_np.shape}")
            auc = 0.0

        if return_details:
            return avg_loss, accuracy, auc, {
                'sample_ids': all_sample_ids,
                'patient_ids': all_patient_ids,
                'probs': all_probs_np,
                'labels': all_labels_np,
            }
        return avg_loss, accuracy, auc

    def _aggregate_patient_results(self, results_df: pd.DataFrame):
        """Aggregate slide-level predictions to patient-level using configured strategy."""
        return self.voting_strategy.aggregate(results_df, self.model_cfg.num_classes)

    def extract_features(self, model, loader, normalize=True):
        """
        Extract bag-level embeddings from the model for all samples in the loader.
        Aggregates slide-level features to patient-level using mean pooling.
        
        Args:
            model: Trained MIL model
            loader: DataLoader containing samples to extract features from
            normalize: If True, apply L2 normalization to features
            
        Returns:
            dict: {
                'patient_ids': list of patient IDs,
                'features': np.ndarray of shape (n_patients, feature_dim),
                'labels': np.ndarray of shape (n_patients,)
            }
        """
        model.eval()
        slide_features = []
        slide_patient_ids = []
        slide_labels = []

        def _find_classifier_linear(module_root: nn.Module) -> nn.Linear | None:
            # Heuristic: pick the last Linear whose out_features matches num_classes (or 1 for binary)
            target_out = 1 if self.model_cfg.num_classes == 1 else int(self.model_cfg.num_classes)
            candidates: list[nn.Linear] = []
            for m in module_root.modules():
                if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == target_out:
                    candidates.append(m)
            return candidates[-1] if candidates else None
        
        with torch.no_grad():
            for raw_batch in tqdm(loader, desc="  Extracting features", leave=False):
                batch = self._move_to_device(raw_batch)
                inputs = self._model_inputs(batch)
                patient_ids = batch.get('patient_id', batch.get('sample_id'))
                labels = batch['label']

                outputs = model(inputs)
                bag_embedding = outputs.get('bag_embeddings')

                if bag_embedding is None:
                    captured: Dict[str, torch.Tensor] = {}

                    classifier_linear = _find_classifier_linear(model)
                    if classifier_linear is None and hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
                        classifier_linear = _find_classifier_linear(getattr(model, "model"))

                    if classifier_linear is None:
                        raise RuntimeError(
                            "Model did not return bag embeddings and no classifier Linear layer was found. "
                            "Please implement embedding output for this model."
                        )

                    def _hook(mod, inp, out):
                        if isinstance(inp, (tuple, list)) and inp and isinstance(inp[0], torch.Tensor):
                            captured['embedding'] = inp[0].detach()

                    handle = classifier_linear.register_forward_hook(_hook)
                    try:
                        _ = model(inputs)
                    finally:
                        handle.remove()

                    if 'embedding' not in captured:
                        raise RuntimeError(
                            "Failed to capture embedding via classifier hook. "
                            "Please check model forward graph."
                        )
                    bag_embedding = captured['embedding']

                slide_features.append(bag_embedding.cpu().numpy())
                slide_patient_ids.extend(patient_ids)
                slide_labels.extend(labels.cpu().numpy())
        
        # Concatenate all slide features
        slide_features = np.concatenate(slide_features, axis=0)  # (n_slides, embed_dim)
        slide_labels = np.array(slide_labels)
        
        # Aggregate to patient level using mean pooling
        patient_features_dict = {}
        patient_labels_dict = {}
        
        for i, pid in enumerate(slide_patient_ids):
            if pid not in patient_features_dict:
                patient_features_dict[pid] = []
                patient_labels_dict[pid] = []
            patient_features_dict[pid].append(slide_features[i])
            patient_labels_dict[pid].append(slide_labels[i])
        
        # Compute mean features per patient
        patient_ids_list = list(patient_features_dict.keys())
        patient_features = np.array([
            np.mean(patient_features_dict[pid], axis=0) for pid in patient_ids_list
        ])
        patient_labels = np.array([
            int(np.bincount(np.array(patient_labels_dict[pid]).astype(int)).argmax())
            for pid in patient_ids_list
        ])
        
        # L2 normalize features (important for fusion with zero-padding)
        if normalize:
            norms = np.linalg.norm(patient_features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid division by zero
            patient_features = patient_features / norms
        
        return {
            'patient_ids': patient_ids_list,
            'features': patient_features,
            'labels': patient_labels
        }

    def train_full_dataset(self):
        """
        Train on full dataset for deployment model (no validation).
        """
        print(f"Training on full dataset ({len(self.dataset)} samples) for deployment...")
        
        train_loader = DataLoader(self.dataset, batch_size=self.training_cfg.batch_size, shuffle=True)
        
        model = self.model_builder().to(self.device)
        
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

            model = self.model_builder().to(self.device)

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

            # Extract and save features for fusion (train + test sets)
            if self.logging_cfg.save_features:
                print(f"  Extracting features for fusion...")
                # Create train loader without shuffle for feature extraction
                train_loader_no_shuffle = DataLoader(train_subset, batch_size=self.training_cfg.batch_size, shuffle=False)
                
                train_features = self.extract_features(model, train_loader_no_shuffle, normalize=True)
                test_features = self.extract_features(model, test_loader, normalize=True)
                
                features_path = os.path.join(self.logging_cfg.save_dir, f'fold_{fold+1}_features.pt')
                torch.save({
                    'train': train_features,
                    'test': test_features,
                    'fold': fold + 1
                }, features_path)
                print(f"  Features saved to {features_path}")

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

