from data_edit import transform_data
from utils import TrafficSignsDataset, get_loader
from model import CNN

import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import wandb

if __name__ == "__main__":
  wandb_init = False
  load_model = False
  checkpoint_path = ""
  save_path = "models"
  transform_data = False 

  print("Editing data...")
  
  if transform_data:
      transform_data("data/Train.csv", "data", "resized_data", type="train")
      transform_data("data/Test.csv", "data", "resized_data", type="test")
  
  transform = transforms.Compose([
    transforms.ColorJitter(0.2, 0.1, 0.0, 0.0),
    transforms.RandomRotation(10),
  ])

  print("Creating dataset...")

  train_dataset = TrafficSignsDataset("resized_data/train-annotations.csv", True)
  test_dataset = TrafficSignsDataset("resized_data/test-annotations.csv", False)

  train_dataloader = get_loader(train_dataset, "resized_data/train-annotations.csv", 64, False, shuffle=True)
  test_dataloader = get_loader(test_dataset, "resized_data/test-annotations.csv", 64, False, shuffle=True)

  # hyperparameters
  in_channels = 3
  num_classes = 43
  learning_rate = 1e-3
  batch_size = 64
  num_epochs = 30
  device = ("cuda" if torch.cuda.is_available() else "cpu")

  if wandb_init:
    model_name = "traffic-signs-cnn"
    wandb.init(project=model_name)
    wandb.config = {
      "learning_rate" : learning_rate,
      "epochs" : num_epochs,
      "bathch_size" : batch_size,
    }

  print("Creating model...")

  model = CNN(num_classes, in_channels).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  if load_model:
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
  
  model.train()

  for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    training_accuracy = []
    training_loss = []
    
    for idx, (features, labels) in enumerate(train_dataloader):
      features = features.to(device).float()
      labels = labels.to(device)
      
      scores = model(features)
      
      loss = criterion(scores, labels)
      num_correct = (scores.argmax(1) == labels).sum().item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      accuracy = num_correct/batch_size
      training_accuracy.append(accuracy)
      training_loss.append(loss.item())

      if wandb_init:
        wandb.log({
          "Training loss" : loss.item(),
          "Training accuracy" : accuracy,
        })

    model.eval()
    testing_accuracy = []
    testing_loss = []

    with torch.no_grad():
      for idx, (features, labels) in enumerate(test_dataloader):
        features = features.to(device).float()
        labels = labels.to(device)

        scores = model(features)

        loss = criterion(scores, labels)
        num_correct = (scores.argmax(1) == labels).sum().item()
        
        accuracy = num_correct/batch_size
        testing_accuracy.append(accuracy)
        testing_loss.append(loss.item())

    avg_training_accuracy = torch.Tensor(training_accuracy).mean().item()
    avg_training_loss = torch.Tensor(training_loss).mean().item()
    avg_testing_accuracy = torch.Tensor(testing_accuracy).mean().item()
    avg_testing_loss = torch.Tensor(testing_loss).mean().item()
        
    print(f"Epoch : {epoch+1}")
    print(f"Training accuracy : {avg_training_accuracy} | Training loss : {avg_training_loss}")
    print(f"Testing accuracy : {avg_testing_accuracy} | Testing loss : {avg_testing_loss}")
    
    metrics =  {
      "Epoch" : epoch+1,
      "Training accuracy" : avg_training_accuracy,
      "Training loss" : avg_training_loss,
      "Testing accuracy" : avg_testing_accuracy,
      "Testing loss" : avg_testing_loss,
    }
    if wandb_init:
      wandb.log(metrics)
    model_param = {
      "model_state_dict" : model.state_dict(),
      "opimizer_state_dict" : optimizer.state_dict(),
    }
    torch.save(model_param, f"models/v2/noaug_dropout{epoch}.pt")
      
