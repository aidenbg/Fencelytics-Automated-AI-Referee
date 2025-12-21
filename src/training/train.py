import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime

"""
FENCING ACTION LSTM TRAINING

Example script showing how to train an LSTM on the prepared fencing data.
Uses 8-frame sequences for action recognition.
Supports both single-label and multi-label classification.

Configuration:
    - Set default paths below or use command line arguments
    - All outputs (model, plots, reports) saved to organized directory structure

Usage:
    python train_lstm.py                    # Standard training
    python train_lstm.py --macoptimize     # With Mac CPU optimizations
    python train_lstm.py --name "exp1"     # Named experiment
    python train_lstm.py --help            # Show all options
"""

# Default configuration - CHANGE THESE AS NEEDED
DEFAULT_DATA_PATH = '/Users/aidenburagohain/Coding/FencingAI/data/pose_data/training_data/combined_data.pkl'  # Your data file
DEFAULT_OUTPUT_DIR = '/Users/aidenburagohain/Coding/FencingAI/data/LSTM/runs'                # Where to save results
DEFAULT_EXPERIMENT_NAME = "lstm_Jul29"                   # Will use timestamp if None

class FencingDataset(Dataset):
    """PyTorch dataset for fencing sequences"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels) if len(labels.shape) > 1 else torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class FencingLSTM(nn.Module):
    """LSTM model for fencing action recognition
    
    Designed for 8-frame sequences - shorter context but more training samples.
    Uses bidirectional LSTM with attention for better temporal modeling.
    """
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, 
                 num_classes=11, dropout=0.3, multi_label=False):
        super(FencingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.multi_label = multi_label
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=dropout
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation for output
        self.output_activation = nn.Sigmoid() if multi_label else nn.Identity()
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch, seq_len=8, hidden*2)
        
        # Apply attention
        lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden*2)
        attn_out, attn_weights = self.attention(
            lstm_out_transposed, 
            lstm_out_transposed, 
            lstm_out_transposed
        )
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden*2)
        
        # Use last timestep output
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.output_activation(out)
        
        return out


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cuda', multi_label=False,
                early_stopping_patience=10):
    """Train the LSTM model with early stopping"""
    
    # Loss and optimizer
    if multi_label:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                if multi_label:
                    # Multi-label accuracy: exact match
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == labels).all(dim=1).sum().item()
                else:
                    # Single-label accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Also print first epoch
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.2f}%')
            if epochs_without_improvement > 0:
                print(f'  Epochs without improvement: {epochs_without_improvement}')
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation loss: {best_val_loss:.4f}')
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, device='cuda', multi_label=False,
                   action_names=None, output_path=None, no_display=False):
    """Evaluate model and generate classification report"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            if multi_label:
                predicted = (outputs > 0.5).cpu().numpy()
                all_predictions.extend(predicted)
                all_labels.extend(labels.numpy())
            else:
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    if multi_label:
        # Multi-label classification report
        print("\nMulti-label Classification Report:")
        print("="*60)
        report_text = "Multi-label Classification Report\n" + "="*60 + "\n"
        
        for i, action in enumerate(action_names):
            print(f"\n{action}:")
            report = classification_report(
                all_labels[:, i], 
                all_predictions[:, i],
                target_names=['Not Present', 'Present']
            )
            print(report)
            report_text += f"\n{action}:\n{report}\n"
        
        # Save report if output path provided
        if output_path:
            report_file = os.path.join(output_path, 'classification_report.txt')
            with open(report_file, 'w') as f:
                f.write(report_text)
            print(f"\nSaved classification report to {report_file}")
            
    else:
        # Single-label classification report
        print("\nClassification Report:")
        print("="*60)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=action_names)
        print(report)
        
        # Save report if output path provided
        if output_path:
            report_file = os.path.join(output_path, 'classification_report.txt')
            with open(report_file, 'w') as f:
                f.write("Classification Report\n")
                f.write("="*60 + "\n")
                f.write(report)
            print(f"\nSaved classification report to {report_file}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=action_names,
                    yticklabels=action_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if output_path:
            cm_file = os.path.join(output_path, 'confusion_matrix.png')
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {cm_file}")
            
            # Also save confusion matrix as text
            cm_text_file = os.path.join(output_path, 'confusion_matrix.txt')
            with open(cm_text_file, 'w') as f:
                f.write("Confusion Matrix\n")
                f.write("Rows: True labels, Columns: Predicted labels\n")
                f.write("Actions: " + ", ".join(action_names) + "\n\n")
                np.savetxt(f, cm, fmt='%d', delimiter='\t')
        
        if not no_display:
            plt.show()
        else:
            plt.close()
    
    return all_predictions, all_labels


# Main training script
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM for fencing action recognition')
    parser.add_argument('--macoptimize', action='store_true', 
                      help='Enable optimizations for Mac CPU training')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                      help='Path to the prepared training data')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs to train (overrides defaults)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Directory to save all outputs')
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                      help='Name for this training run (creates subfolder)')
    parser.add_argument('--no-display', action='store_true',
                      help='Do not display plots (useful for remote training)')
    args = parser.parse_args()
    
    # Create output directory structure
    if args.name is None:
        # Generate name with timestamp if not provided
        args.name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_path = os.path.join(args.output_dir, args.name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Run: {args.name}")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Will save:")
    print(f"  - Model: fencing_lstm_model.pth")
    print(f"  - Training curves: training_curves.png")
    print(f"  - Confusion matrix: confusion_matrix.png")
    print(f"  - Classification report: classification_report.txt")
    print(f"  - Training history: training_history.json")
    print(f"  - Configuration: training_config.json")
    print(f"{'='*60}\n")
    
    # Load prepared data
    print(f"Loading training data from: {args.data_path}")
    try:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"\nError: Could not find data file at {args.data_path}")
        print("Make sure you've run prepare_training_data.py first!")
        return
    
    sequences = data['sequences']
    labels = data['labels']
    action_mapping = data['action_mapping']
    
    # Determine if multi-label
    multi_label = len(labels.shape) > 1
    
    print(f"Data shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Multi-label: {multi_label}")
    
    if args.macoptimize:
        print("\n" + "="*50)
        print("MAC OPTIMIZATION MODE ENABLED")
        print("="*50)
        print("Optimizations:")
        print("  - Batch size: 16 (instead of 32)")
        print("  - Epochs: 30 (instead of 50)")
        print("  - Single-threaded processing")
        print("  - These settings optimize for CPU performance")
        print("="*50 + "\n")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Val: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Create data loaders
    if args.macoptimize:
        batch_size = 16  # Smaller batch for CPU
        num_workers = 0  # Single thread for Mac CPU
        print("\nMac optimizations enabled:")
        print("  - Batch size: 16")
        print("  - Single-threaded DataLoader")
    else:
        batch_size = 32
        num_workers = 0  # Default to 0 for stability
    
    train_dataset = FencingDataset(X_train, y_train)
    val_dataset = FencingDataset(X_val, y_val)
    test_dataset = FencingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Apply Mac-specific optimizations if requested
    if args.macoptimize and device.type == 'cpu':
        print("Applying Mac CPU optimizations...")
        torch.set_num_threads(1)  # Can experiment with more threads
        print("  - Limited to single thread")
    
    model = FencingLSTM(
        input_size=51,
        hidden_size=128,
        num_layers=2,
        num_classes=len(action_mapping),
        dropout=0.3,
        multi_label=multi_label
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    if args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = 30 if args.macoptimize else 50
    
    if args.macoptimize:
        print(f"  - Mac optimizations enabled")
    print(f"  - Training for {num_epochs} epochs")
    
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=num_epochs, 
        learning_rate=0.001,
        device=device,
        multi_label=multi_label
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    
    # Save training curves
    curves_file = os.path.join(output_path, 'training_curves.png')
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved training curves to {curves_file}")
    if not args.no_display:
        plt.show()
    else:
        plt.close()
    
    # Save training history data
    history_file = os.path.join(output_path, 'training_history.json')
    history_data = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_accuracies': [float(x) for x in val_accuracies],
        'epochs': len(train_losses)
    }
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history to {history_file}")
    
    # Evaluate on test set
    action_names = [name for name, _ in sorted(action_mapping.items(), 
                                               key=lambda x: x[1])]
    print("\nEvaluating on test set...")
    predictions, true_labels = evaluate_model(
        model, test_loader, device, multi_label, action_names, output_path, args.no_display
    )
    
    # Save model
    model_file = os.path.join(output_path, 'fencing_lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'action_mapping': action_mapping,
        'multi_label': multi_label,
        'model_config': {
            'input_size': 51,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': len(action_mapping),
            'dropout': 0.3
        }
    }, model_file)
    
    print(f"\nModel saved as {model_file}")
    
    # Save training configuration
    config_file = os.path.join(output_path, 'training_config.json')
    config_data = {
        'data_path': args.data_path,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_shape': list(sequences.shape),
        'multi_label': multi_label,
        'mac_optimized': args.macoptimize,
        'timestamp': datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved training configuration to {config_file}")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    
# Usage examples:
# python train_lstm.py                          # Standard training (saves to ./training_results/run_TIMESTAMP/)
# python train_lstm.py --macoptimize           # Mac optimized (30 epochs)
# python train_lstm.py --macoptimize --epochs 40  # Mac optimized with 40 epochs
# python train_lstm.py --name "experiment_1"   # Save to ./training_results/experiment_1/
# python train_lstm.py --output-dir ./models --name "final_model"  # Custom output location
# python train_lstm.py --no-display            # Don't show plots (for remote/automated training)
# python train_lstm.py --data-path ./my_data.pkl --epochs 100  # Custom data and epochs