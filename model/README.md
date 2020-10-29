# Model part
- The folder in model are organized as follow:
```
.
├── backbone
│   ├── alexnet.py
│   ├── resnet.py
│   └── utils
│       ├── alexnet_torch.py
│       └── resnet_torch.py
├── loss
│   └── triplet_loss.py
├── model
│   └── triplet_model.py
└── README.md
```

- The backbone folder contains the backbone model of out net, We copied the source code of model from torch vision with minor modify for further develope.

- The loss folder contains the wrapped triplet loss function.

- The model folder implemented various model framwork.

