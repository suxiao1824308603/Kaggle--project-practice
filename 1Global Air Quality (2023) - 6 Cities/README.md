# This project is Data analyze of [Global Air Quality](https://www.kaggle.com/code/devraai/global-air-quality-data-analysis-and-aqi-predic/input?select=London_Air_Quality.csv).

In the code, we achieve the data mining first, and further predict the AQI of different city based on MultiscaleCNN + ResidualLSTM.

## MultiScaleCNN Module is shown as follow,
```python
  class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim):
      super().__init__()
      self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=(0,0))
      self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1))
      self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,5), padding=(0,2))
      self.branch4 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,7), padding=(0,3))
      self.relu = nn.ReLU()

    def forward(self, x):
      x1 = self.branch1(x)
      x2 = self.branch2(x)
      x3 = self.branch3(x)
      x4 = self.branch4(x)
      out = torch.cat([x1, x2, x3, x4], dim=1)
      return self.relu(out)
```
## The predictions are as follows (Sydney, London, NewYork, Brasilia)
### Sydney

![Pre](./image/Sydney.png){:height="50%" width="70%"}

<div align="center">
  <img src"./image/Sydney.png" width="300">
</div>

![Sydney2](image/Pair_analyze.png)

London-Prediction ![London](image/London.png)

NewYork-Prediction ![NewYork](image/New_York.png)
