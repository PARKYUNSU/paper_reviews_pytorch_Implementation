import torch

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pred = model(X, y_input, tgt_mask)
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list.append(train_loss)

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list.append(validation_loss)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

    return train_loss_list, validation_loss_list