## Continual Federated Learning with Diffusion

NeurIPS 2024, Submission 5599

-----------------------------------------------------------------------------------------------

## Requirements
Please download packages via `pip install -r requirements.txt` or below
```
* python == 3.10.12
* einops==0.7.0
* labml-helpers==0.4.89
* labml-nn==0.4.136
* numpy==1.23.5
* torch==2.2.2
* torchvision==0.17.2
```

## Dataset

For the MNIST, Fashion-MNIST, and CIFAR-10 datasets, they can all be downloaded through torchvision.datasets. You don't need to do anything! 
For the PACS dataset, please refer to the official download link in the original PACS dataset paper [https://dali-dl.github.io/project_iccv2017.html](https://dali-dl.github.io/project_iccv2017.html).

## Files
```
DCFL/
│   README.md
│   requirements.txt    
│
└─── CFL_CD/
    │   CFL_CD.py
    │   load_dataset.py
    │   models.py
    │   options.py
    │
    └─── data/
        └─── PACS/ -- Put the downloaded PACS dataset here
│
└─── MNIST/
    │   {args.dataset}/CD_{args.task}_CFL_{args.dataset}_{args.framework}.npy -- saved results here
    └─── model/
        │   Diffusion_{args.task}_CFL_{args.dataset}_{args.framework}.pth -- saved models here


