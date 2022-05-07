# Individual Models

For the following tutorial, suppose that we are using `Config`, `criterion`, and randomly generated data as follows:  

```python
from coperception.models.det import *
from coperception.utils.CoDetModule import FaFModule
from coperception.configs import Config
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
import torch.optim as optim

import numpy as np
import torch

batch_size = 1
num_agent = 6
width = 256
height = 256
num_channels = 13
collaboration_layer = 3
learning_rate = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_shapes = {
        'bev_seq': (num_agent * batch_size, 1, width, height, num_channels),
        'labels': (num_agent * batch_size, width, height, num_agent, 2),
        'reg_targets': (num_agent * batch_size, width, height, num_agent, 1, num_agent),
        'anchors': (num_agent * batch_size, width, height, num_agent, num_agent),
        'reg_loss_mask': (num_agent * batch_size, width, height, num_agent, 1),
        'vis_maps': (num_agent * batch_size, 0),
        'target_agent_ids': (batch_size, num_agent),
        'num_agent': (batch_size, num_agent),
        'trans_matrices': (batch_size, num_agent, num_agent, 4, 4)
}

# randomly generated data
data = {
        'bev_seq': torch.rand(*data_shapes['bev_seq']).to(device),
        'labels': torch.rand(*data_shapes['labels']).to(device),
        'reg_targets': torch.rand(*data_shapes['reg_targets']).to(device),
        'anchors': torch.rand(*data_shapes['anchors']).to(device),
        'reg_loss_mask': torch.from_numpy(np.random.choice(a=[False, True], size=(data_shapes['reg_loss_mask']))).to(device),
        'vis_maps': torch.rand(*data_shapes['vis_maps']).to(device),
        'target_agent_ids': torch.from_numpy(np.random.choice(a=[i for i in range(num_agent)], size=(data_shapes['target_agent_ids']))).to(device),
        'num_agent': torch.from_numpy(np.random.choice(a=[i for i in range(num_agent)], size=(data_shapes['num_agent']))).to(device),
        'trans_matrices': torch.from_numpy(np.random.choice(a=[i for i in range(num_agent)], size=(data_shapes['trans_matrices']))).to(device)
}

config = Config('train', binary=True, only_det=True)
criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}
```
<br> 

## FaFNet
::: coperception.models.det.FaFNet.FaFNet
    selection:
      members: none

![FaFNet](../assets/images/fafnet.png)

### Detection
**Initialization**
```python
model = FaFNet(
        config, 
        layer=collaboration_layer, 
        kd_flag=False, 
        num_agent=num_agent
)

config.flag = 'lowerbound' # [lowerbound / upperbound]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

faf_module = FaFModule(
        model=model,
        teacher=None,
        config=config, 
        optimizer=optimizer,
        criterion=criterion, 
        kd_flag=False
)
```

**Training**
```python
loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
```

**Testing**
```python
faf_module.model.eval()

checkpoint = torch.load('/path/to/checkpoint/file.pth')
faf_module.model.load_state_dict(checkpoint['model_state_dict'])
faf_module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
faf_module.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

loss, cls_loss, loc_loss, result = faf_module.predict_all(data, batch_size=1, num_agent=num_agent)
```


<br> 