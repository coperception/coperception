# Collaborative Models

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

## DiscoNet

::: coperception.models.det.DiscoNet.DiscoNet
    selection:
      members: none

![DiscoNet](./assets/images/disconet.png)


For DiscoNet, we need a pre-trained model as the teacher net in the knowledge distillation process.  
In the DiscoNet paper above, we used `FaFNet` as the teacher net.  
Before training DiscoNet, we need to train `FaFNet` and save its weights.  
In the following example code, we load the weights to `TeacherNet` and use it as the teacher model to train `DiscoNet`.  
`TeacherNet` has the same architecture as `FaFNet`. You can checkout its implementation here:  
::: coperception.models.det.TeacherNet.TeacherNet
    selection:
      members: none

<br/>

### Detection
**Initialization**
```python
teacher = TeacherNet(config)  # Teacher model for DiscoNet

checkpoint_teacher = torch.load('/path/to/faf_net/checkpoint/file.pth')
teacher.load_state_dict(checkpoint_teacher['model_state_dict'])
teacher.eval() # Put in evaluation mode. TeacherNet is already trained.

model = DiscoNet(
        config, 
        layer=collaboration_layer, 
        kd_flag=True, 
        num_agent=num_agent
)

config.flag = 'disco'
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

faf_module = FaFModule(
        model=model,
        teacher=teacher,
        config=config, 
        optimizer=optimizer,
        criterion=criterion, 
        kd_flag=True
)
```

**Training**
```python
loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
```

**Testing**
```python
loss, cls_loss, loc_loss, result, save_agent_weight_list = fafmodule.predict_all(data, batch_size=1, num_agent=num_agent)
```

<br>
## V2VNet
::: coperception.models.det.V2VNet.V2VNet
    selection:
      members: none

![V2VNet](./assets/images/v2vnet.png)

### Detection
**Initialization**
```python
model = V2VNet(
        config, 
        gnn_iter_times=3, 
        layer=collaboration_layer, 
        layer_channel=256, 
        num_agent=num_agent
)

config.flag = 'v2v'
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

## When2com / Who2com
::: coperception.models.det.When2com.When2com
    selection:
      members: none

![When2com](./assets/images/when2com.png)

When2com and Who2com uses the same model.  
They only differ in the inference stage.

### Detection
**Initialization**
``` python
model = When2com(
        config, 
        layer=collaboration_layer,
        warp_flag=True,
        num_agent=num_agent
)

config.flag = 'when2com'
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

faf_module = FaFModule(
        model=model,
        teacher=None,
        config=config, 
        optimizer=optimizer,
        criterion=criterion, 
        kd_flag=False
)

loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
```

**Training**
```python
loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
```

**Testing**  
**When2com**
```python
config.inference = 'activated'
loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
``` 

**Who2com**
```python
config.flag = 'who2com'
config.inference = 'argmax_test'
loss, cls_loss, loc_loss = faf_module.step(data, batch_size, num_agent=num_agent)
```