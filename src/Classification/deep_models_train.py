import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from Classification.deep_models_init import initialize_model


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for tabular deep classifiers.
    """

    def __init__(
        self,
        model_name="MLP",
        max_epochs=120,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5,
        val_size=0.2,
        patience=12,
        min_delta=1e-4,
        early_stopping=True,
        random_state=0,
        verbose=False,
    ):
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_size = val_size
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def _to_numpy(self, x):
        if hasattr(x, "to_numpy"):
            return x.to_numpy()
        return np.asarray(x)

    def fit(self, X, y):
        x_np = self._to_numpy(X).astype(np.float32)
        y_np = self._to_numpy(y)

        self.classes_, y_encoded = np.unique(y_np, return_inverse=True)
        y_encoded = y_encoded.astype(np.int64)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if torch.cuda.is_available():
            self.device_ = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
        else:
            self.device_ = torch.device("cpu")

        x_tensor = torch.from_numpy(x_np).to(self.device_)
        y_tensor = torch.from_numpy(y_encoded).to(self.device_)

        train_idx = np.arange(x_np.shape[0])
        val_idx = None

        if self.val_size > 0.0 and x_np.shape[0] > 10:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
                train_idx, val_idx = next(splitter.split(x_np, y_encoded))
            except ValueError:
                # Fallback for edge cases where stratification is not feasible.
                rng = np.random.default_rng(self.random_state)
                shuffled = np.arange(x_np.shape[0])
                rng.shuffle(shuffled)
                val_count = max(1, int(np.floor(self.val_size * x_np.shape[0])))
                val_idx = shuffled[:val_count]
                train_idx = shuffled[val_count:]

        train_dataset = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val_idx is not None and len(val_idx) > 0:
            val_dataset = TensorDataset(x_tensor[val_idx], y_tensor[val_idx])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        input_dim = x_np.shape[1]
        n_classes = len(self.classes_)
        self.model_ = initialize_model(self.model_name, input_dim, n_classes).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.n_features_in_ = input_dim

        self.model_.train()
        best_val_loss = np.inf
        stale_epochs = 0
        best_state_dict = None
        self.best_epoch_ = 0

        for epoch in range(int(self.max_epochs)):
            epoch_train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_train_loss += float(loss.item())

            val_loss = epoch_train_loss
            if val_loader is not None:
                self.model_.eval()
                running_val = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits = self.model_(xb)
                        running_val += float(criterion(logits, yb).item())
                val_loss = running_val
                self.model_.train()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[{self.model_name}] epoch={epoch + 1} train_loss={epoch_train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )

            if self.early_stopping:
                if val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = val_loss
                    stale_epochs = 0
                    self.best_epoch_ = epoch + 1
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(self.patience):
                        break

        if self.early_stopping and best_state_dict is not None:
            self.model_.load_state_dict(best_state_dict)

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        x_np = self._to_numpy(X).astype(np.float32)
        x_tensor = torch.from_numpy(x_np).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_tensor)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.classes_[pred_idx]

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        x_np = self._to_numpy(X).astype(np.float32)
        x_tensor = torch.from_numpy(x_np).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
