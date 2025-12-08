import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.models.mlp import MLP


def parse_args():
	parser = argparse.ArgumentParser(description="Train MLP model")
	parser.add_argument("--input-dim", type=int, required=True,
						help="Input feature dimension")
	parser.add_argument("--num-classes", type=int, required=True,
						help="Number of output classes")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--model-path", type=str, default="mlp.pth",
						help="Path to save trained model")
	return parser.parse_args()


def create_dummy_dataloader(input_dim: int, num_classes: int, batch_size: int) -> DataLoader:
	# Placeholder: replace with real dataset loading logic used in the project.
	num_samples = 1024
	x = torch.randn(num_samples, input_dim)
	y = torch.randint(0, num_classes, (num_samples,))
	dataset = TensorDataset(x, y)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in dataloader:
		inputs = inputs.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / total
	epoch_acc = 100.0 * correct / total
	return epoch_loss, epoch_acc


def main():
	args = parse_args()

	device = torch.device(args.device)

	model = MLP(input_dim=args.input_dim, num_classes=args.num_classes).to(device)
	train_loader = create_dummy_dataloader(args.input_dim, args.num_classes, args.batch_size)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, criterion, optimizer, device
		)
		print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}%")

	torch.save(model.state_dict(), args.model_path)
	print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
	main()

