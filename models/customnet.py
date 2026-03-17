from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Add more layers...
        self.fc1 = nn.Linear(256*28*28, 200) # 200 is the number of classes in TinyImageNet
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224 immagine iniziale dopo 3 pool diventa 28 * 28

        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool(x) # B x 64 x 112 x 112
        x = self.conv2(x).relu() # B x 128 x 224 x 224
        x = self.pool(x) # B x 128 x 112 x 112
        x = self.conv3(x).relu() # B x 256 x 224 x 224
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        return x
