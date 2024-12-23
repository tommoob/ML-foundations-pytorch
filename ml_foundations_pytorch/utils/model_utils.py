import torch
import matplotlib.pyplot as plt


def run_inference(model, test_loader, device="cpu"):
    model = model.to(device)
    all_predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions


def display_sample_predictions(model, test_loader, device="cpu"):
    model = model.to(device)

    # Display a few predictions
    how_many_sets, count = 4, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Plot the first batch
            for i in range(8):  # Display the first 8 images
                plt.subplot(2, 4, i + 1)
                plt.imshow(images[i].cpu().squeeze(), cmap="gray")
                plt.title(f"Pred: {predictions[i].item()}\nTrue: {labels[i].item()}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
            count += 1
            if count > how_many_sets:
                break
