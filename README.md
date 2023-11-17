# Pytorch_Day1
This is my Diary to be 3D object Detection professional Developer

# import the libaries for making the model 
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import transforms 

# Define the data for using the models 
training_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Set the batch_size 
batch_size = 64

# define the dataLoader 
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

for X, y in test_dataloader :
    print(f"Shape of X[N, C, H, W]:{X.shape}")
    print(f"Shape of y: y.shape")
    break 

# Setting the device 
DEVICE = (
    "cude" 
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), 
        )
    def forward(self, x) :
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(DEVICE) 
print(model)
    
# Define the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)\
    
def train(dataloader, model, loss_fn, optimizer ):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(DEVICE) , y.to(DEVICE)
        
        # Calculate the loss this model 
        pred = model(X) # Same Error massage that is cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not tuple -> need to be fixed
        train_loss = loss_fn(pred, y) 
        
        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            train_loss, current = train_loss.item(), batch*len(X)
            print(f"loss: {train_loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn): # optmizer is not used in the testing
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() 
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size 
        
        print(f"Test Error \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Train the model -> epoch, model, loss_fn, optimizer, train_dataloader need to be defined
epochs = 5
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save the model 
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Lpad the model
model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth"))

# predict the model 
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() 
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(DEVICE)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: '{predicted}', Actual: '{actual}'")
