import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()