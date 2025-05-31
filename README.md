# Once-for-All: Train One Network And Specialize It For Efficient Deployment

## 0. Introduction

  Neural Architecture Search (NAS) methods have shown great promise in automating the design of neural networks. However, most NAS methods are highly resource-intensive, requiring massive computation and retraining for different hardware environments.

 To tackle this challenge, the **Once-for-All (OFA)** network proposes a single supernet that supports diverse architectural configurations, including different kernel sizes, network depths, and widths.

 This supernet is trained once and allows fast specialization to any deployment constraint without the need for retraining. The key idea is to decouple model training from architecture search, enabling efficient and hardware-aware specialization.

> [Figure 1] Conceptual overview of OFA supernet training and subnet extraction.  
![](/imgs/overview.png)

---

## 1. Method

This repository implements the core methodology of OFA up to *progressive shrinking* for kernel size, depth, and width elasticity. Hardware-specific adaptation (Section 3.4 & 4.2 in the paper) is **not** included.

### 1.1. Search Space

The OFA search space is based on MobileNetV3-like structures, comprising:

- Elastic kernel(*k*) sizes: varying size of the kernel for the convolution operatrion.
- Elastic depths(*d*): varying number of layers in each block
- Elastic widths(*w*): adjusting channel expansion ratios


Each block (Mobile Inverted Bottleneck Convolution, MBConv) can adapt its:

- Kernel size
- Depth (number of layers)
- Width (output channels)



Each elastic operation is progressively shrunk during training.

### 1.2. Progressive Shrinking

Instead of training every possible sub-network separately, the OFA approach uses **progressive shrinking**:

1. **Train for maximum configuration** (largest kernel size, depth, width)
2. **Shrink one dimension at a time**, gradually expanding support for more subnetworks:
   - Shrink kernel size using kernel transfor matrices.
   - Shrink depth by keeping the first d layers.
     > [Figure 2] Progressively shrinking depth. 
     ![](/imgs/Elastic_depth.PNG)
   - Shrink width by reorganizng models channels according to their importance (in this case, L1-norm)
     > [Figure 3] Progressively shrinking width.
     ![](/imgs/Elastic_width.PNG)

Thus, the supernet becomes capable of supporting many subnetworks efficiently.

In this codebase:

- `ProgressiveShrinking()` applies a given config (combination of {k, d, w}) to the supernet and extractes the subnet via `get_fixed_Subnet()`.


> [Figure 4] Progressive Shrinking Procedure.
![](/imgs/PS.png)

---

## 2. Implementation

### 2.1. Main components

The implementation includes:

- **ElasticConv**:
 Convolution layers with elastic kernel size and width support, applying kernel transformations for different sizes.

> [Figure 5] Architecture of an ElasticConv.
![](/imgs/Elastic_Conv.png)
- **ElasticSqueezeAndExcite**:
  Squeeze-and-excite operation with elastic input channel support.

> [Figure 6] Architecture of an ElasticSqueezeAndExcite.
![](/imgs/Elastic_SE.png)
- **ElasticMBblock**:
  MobileNetV3-like blocks with depthwise separable convolutions and Squeeze-and-Excitation (SE) modules.

> [Figure 7] Architecture of an ElasticMBblock.
![](/imgs/Elastic_MBblock.png)
- **Kerenl transform matrix**:
  Changes ElasticMBblocks kernel size to support elastic kernel size.

> [Figure 8] Kernel transforming Procedure.
![](/imgs/Kernel_Transform_Matrix.png)
- **ElasticUnit**:
  A stack of ElasticMBblocks, supporting elastic depth.

- **OFAnet**:
  Full supernet with stem, units, final layer, global pooling, and classifier.

> [Figure 9] Architecture of an OFAnet.
![](/imgs/macro_structure.png)
- **ProgressiveShrinking**:
  Shrink the supernet following the given config and return the fixed subnet.

- **Subnet**:
  A Fixed lightweight model excluding unnecessary parts(kernel transfor matrices + etc...) of OFAnet 

Implementation notes:

- Residual connection is used iff the block's channel expansion ratio is 1
- Out channel of the last MBblock is fixed to use the same classifier for all subnets.


[Table 1] Model Components and Functions
> | Component | Description |
> |-----------|-------------|
> | ElasticConv | Kernel elasticity |
> | ElasticSqueezeAndExcite | Elastic squeeze-and-excite operation |
> | ElasticMBblock | Block with elastic depthwise conv |
> | ElasticUnit | Group of elastic blocks |
> | OFAnet | Supernet structure |
> | Subnet | Fixed subnet extracted from supernet |

### 2.2. Experiments

Experiments were conducted on CIFAR-10 using a reduced OFA supernet.

- 1. Setting
	- Dataset : CIFAR-10
	- Config space : [[5, 7], [2, 3], [2, 4]]
	- Supernet : OFAnet(num_units=1, in_channels=3, num_classes=10, max_width=4, max_depth=3, max_kernel_size=7)
	- loss : Cross entropy

- 2. Train the full model(supernet)
	- Train for total 180 epochs.
	- SGD optimizer with momentum of 0.9 and weight decay of 3e-5.
	- Cosine schedule is used with initial learning rate of 2.6

	This part is implemented using `train_one_epoch()` function.

- 3. Shrink the model and fine-tune both small and big networks.
	1. Elastic kernel size
		- Sample subnet for fixed depth and width
		- Train for total 25 epochs.
		- Same optimizer used for supernet but only targeting the kernel. transfor matrices
		- Cosine schedule with initial learning rate of 0.96.
	2. Elastic depth
		- Sample subnet for fixed width
		- Train for total 25 epochs.
		- Same optimizer used for supernet.
		- Cosine schedule with initial learning rate of 0.08
	3. Elastic width
		- Sample subnet from config space.
		- Rest of the training process is identical with elastic depth part.

	This part is implemented as `train_OFAnet()` function.

- 4. Observing accuracies
	 - Iterate through the config space and observe MACs, Accuracy, Accuracy #25.
	 - Subnet is fine-tuned for 25 epochs using SGD optimizer and cosine scheduler with initial learning rate of 2e-5.
	This part is implemented using `iterate_config_space()`, `evaluate()`, `train_one_epoch()` functions and `calflops` library.

Note: While training the subnet, knowledge distillation is omitted due to computational limit.

## 3. Results

 [Table 2] MACs, and accuracies of subnets for different configs
> | Config: (k, d, w) | MACs | Accuracy | Accuracy #25 |
> |:---:|:---:|:---:|:---:|
> | (5, 2, 2) | 281.76M | 0.1003 | 0.1551 |
> | (5, 2, 4) | 297.26M | 0.1011 | 0.1759 |
> | (5, 3, 2) | 286.95M | 0.0947 | 0.1005 |
> | (5, 3, 4) | 391.43M | 0.1006 | 0.1021 |
> | (7, 2, 2) | 281.91M | 0.1058 | 0.1736 |
> | (7, 2, 4) | 297.75M | 0.1213 | 0.1666 |
> | (7, 3, 2) | 287.29M | 0.098 | 0.1 |  
> | (7, 3, 4) | 393.5M | 0.1126 | 0.1015 |


Note: Accuracy #25 denotes the accuracy after fine-tuning the subnet for 25 epochs.

---

## 4. References

1. [Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/pdf/1908.09791)

2. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)

3. [Searching for MobileNetV3](https://arxiv.org/abs/1908.09791)

4. [Calflops Library](https://github.com/MrYxJ/calculate-flops.pytorch)


---
