import torch 
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = 'model.pt'
SCRIPTED_MODEL_PATH = 'model_scripted.pt'

device=torch.device('cpu')

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=2, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x

def get_model():
    model = M5(n_input=1, n_output=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

if __name__ == '__main__':
    model = M5(n_input=1, n_output=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])
    model.to(device)
    model.eval()

    model_scripted = torch.jit.script(model)

    model_scripted.save(SCRIPTED_MODEL_PATH)
