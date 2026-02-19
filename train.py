import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import ComputeDataAAPL, create_sequences, split_sets
from dataset import MarketDataset
from model import TransformerClassifier


def run_experiment(model_config, training_config):
    # first thing use gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # get the data from the template
    window = training_config["window"]
    seq_len = training_config["seq_len"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]

    # prepare data
    df, model_df = ComputeDataAAPL(window=window)
    X, y = create_sequences(model_df, seq_len=seq_len)
    X_train, y_train, X_val, y_val = split_sets(X, y)

    print("Train positive rate:", y_train.mean())
    print("Val positive rate:", y_val.mean())


    # normalize the train features
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)

    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val - mean) / (std + 1e-8)

    train_dataset = MarketDataset(X_train, y_train)
    val_dataset = MarketDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # init the model
    model = TransformerClassifier(
        feature_dim=model_config["feature_dim"], # how many features there is in that model_df
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"]
    )
    model.to(device)

    # loss and optimizer
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / y_train.sum()],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training

    for epoch in range(epochs):

        # Training Phase

        model.train()
        total_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_pred_ones = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # weighted loss sum
            bs = y_batch.size(0)
            total_loss_sum += loss.item() * bs

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5)

            train_correct += (preds == y_batch.bool()).sum().item()
            train_total += bs
            train_pred_ones += preds.sum().item()

        avg_train_loss = total_loss_sum / train_total
        train_acc = train_correct / train_total
        train_pred_rate = train_pred_ones / train_total

        # val phase

        model.eval()
        val_loss_sum = 0.0
        val_pred_ones = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)

                bs = y_batch.size(0)
                val_loss_sum += loss.item() * bs

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5)

                val_correct += (preds == y_batch.bool()).sum().item()
                val_total += bs
                val_pred_ones += preds.sum().item()

        avg_val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_pred_rate = val_pred_ones / val_total

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val pred+ rate: {val_pred_rate:.3f}"
        )
                
if __name__ == "__main__":
    pass