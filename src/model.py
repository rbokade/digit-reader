import torch
import torch.jit
import torch.nn as nn


class Model(torch.jit.ScriptModule):
# CHECKPOINT_FILENAME_PATTERN = 'model-{}.pth'

    __constants__ = [
        '_hidden1', '_hidden2', '_hidden3', '_hidden4', '_hidden5',
        '_hidden6', '_hidden7', '_hidden8', '_hidden9', '_hidden10',
        '_features', '_classifier',
        '_digit_length', '_digit1', '_digit2', '_digit3', '_digit4', '_digit5'
    ]

    def __init__(self):
        super(Model, self).__init__()

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        self._hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @torch.jit.script_method
    def forward(self, x):
        with torch.no_grad():
            x = self._hidden1(x)
            x = self._hidden2(x)
            x = self._hidden3(x)
            x = self._hidden4(x)
            x = self._hidden5(x)
            x = self._hidden6(x)
            x = self._hidden7(x)
            x = self._hidden8(x)
            x = x.view(x.size(0), 192 * 7 * 7)
            x = self._hidden9(x)
            x = self._hidden10(x)

            length_logits = self._digit_length(x)
            digit1_logits = self._digit1(x)
            digit2_logits = self._digit2(x)
            digit3_logits = self._digit3(x)
            digit4_logits = self._digit4(x)
            digit5_logits = self._digit5(x)

            length_prediction = length_logits.max(1)[1]
            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]
            digit3_prediction = digit3_logits.max(1)[1]
            digit4_prediction = digit4_logits.max(1)[1]
            digit5_prediction = digit5_logits.max(1)[1]

            return torch.stack(
                [length_prediction, digit1_prediction, digit2_prediction, \
                 digit3_prediction, digit4_prediction, digit5_prediction]
            )

    def load_model(self, path_to_checkpoint_file):
        self.load_state_dict(
            torch.load(path_to_checkpoint_file,
            map_location=self.device), strict=False
        )




