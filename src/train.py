import torch
from tqdm.auto import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    model_save_path,
    use_mixed_precision=True,
):
    """
    Train the model.
    """
    best_val_acc = 0.0
    best_val_loss = float("inf")
    min_delta = 1e-5
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    if device.type == "cuda" and use_mixed_precision:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None
        use_mixed_precision = False

    # Early stopping parameters
    patience = 10
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Learning rate for this epoch (for logging)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_mixed_precision and scaler is not None:
                with torch.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        scheduler.step(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.autocast(
                    device_type=device.type, enabled=use_mixed_precision
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": val_loss / ((batch_idx + 1) * inputs.size(0)),
                        "acc": val_correct / val_total,
                    }
                )

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        # Improved early stopping logic
        if val_epoch_loss < (best_val_loss - min_delta):
            best_val_loss = val_epoch_loss
            early_stop_counter = 0
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"Best model saved! Acc: {best_val_acc:.4f}, "
                    f"Loss: {val_epoch_loss:.4f}"
                )
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step(val_epoch_loss)

    return history
