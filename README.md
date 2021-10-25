# noise-validation

Some **Noisy-label Learning**, **Robust Learning**, **Semi-supervised Learning** and **Training tricks** implementation.

## Algorithms

### Baseline

*MNIST* and *CIFAR10* dataset baseline without label noise or training tricks.

### QBC-Loss

Referring to Active Learning. Using several models to inference, calculating each sample's weighted average loss

### O2U-Net

Title: *O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks*  
Paper: [https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)

### Knowledge distilling

Title: *Distilling the Knowledge in a Neural Network*  
Paper: [https://arxiv.org/pdf/1503.02531.pdf](https://arxiv.org/pdf/1503.02531v1.pdf)

### Mean-Teacher

Title: *Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning
results*  
Paper: [https://arxiv.org/pdf/1703.01780.pdf](https://arxiv.org/pdf/1703.01780v6.pdf)

### Variance

Title: *Active Bias: Training More Accurate Neural Networks by Emphasizing High Variance Samples*  
Paper: [https://arxiv.org/pdf/1704.07433.pdf](https://arxiv.org/pdf/1704.07433v4.pdf)

### Decoupling

Title: *Decoupling “when to update” from “how to update”*  
Paper: [https://arxiv.org/pdf/1706.02613.pdf](https://arxiv.org/pdf/1706.02613v2.pdf)

### MixUp

Title: *mixup: BEYOND EMPIRICAL RISK MINIMIZATION*  
Paper: [https://arxiv.org/pdf/1710.09412.pdf](https://arxiv.org/pdf/1710.09412v2.pdf)

### MentorNet (Haven't done)

Title: *MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels*  
Paper: [https://arxiv.org/pdf/1712.05055.pdf](https://arxiv.org/pdf/1712.05055v2.pdf)

### Co-teaching

Title: *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*  
Paper: [https://arxiv.org/pdf/1804.06872.pdf](https://arxiv.org/pdf/1804.06872v3.pdf)

### Truncated Loss(L<sub>q</sub> Loss, GCE)

Title: *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*  
Paper: [https://arxiv.org/pdf/1805.07836v4.pdf](https://arxiv.org/pdf/1805.07836v4.pdf)

### Forgetting

Title: *AN EMPIRICAL STUDY OF EXAMPLE FORGETTING DURING DEEP NEURAL NETWORK LEARNING*  
Paper: [https://arxiv.org/pdf/1812.05159.pdf](https://arxiv.org/pdf/1812.05159v3.pdf)

### Meta-Weight-Net (Haven't done)

Title: *Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting*  
Paper: [https://arxiv.org/pdf/1902.07379.pdf](https://arxiv.org/pdf/1902.07379v6.pdf)

### MixMatch

Title: *MixMatch: A Holistic Approach to Semi-Supervised Learning*  
Paper: [https://arxiv.org/pdf/1905.02249v1.pdf](https://arxiv.org/pdf/1905.02249v1.pdf)

### SCE

Title: *Symmetric Cross Entropy for Robust Learning with Noisy Labels*  
Paper: [https://arxiv.org/pdf/1908.06112.pdf](https://arxiv.org/pdf/1908.06112v1.pdf)

### NLNL

Title: *NLNL: Negative Learning for Noisy Labels*  
Paper: [https://arxiv.org/pdf/1908.07387.pdf](https://arxiv.org/pdf/1908.07387v1.pdf)

### SELF

Title: *SELF: LEARNING TO FILTER NOISY LABELS WITH SELF-ENSEMBLING*  
Paper: [https://arxiv.org/pdf/1910.01842.pdf](https://arxiv.org/pdf/1910.01842v1.pdf)

### FixMatch

Title: *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*  
Paper: [https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf)

### DivideMix (Haven't done)

Title: *DIVIDEMIX: LEARNING WITH NOISY LABELS AS SEMI-SUPERVISED LEARNING*  
Paper: [https://arxiv.org/pdf/2002.07494v1.pdf](https://arxiv.org/pdf/2002.07494v1.pdf)

### Flooding

Title: *Do We Need Zero Training Loss After Achieving Zero Training Error?*  
Paper: [https://arxiv.org/pdf/2002.08709.pdf](https://arxiv.org/pdf/2002.08709v1.pdf)

### APL

Title *Normalized Loss Functions for Deep Learning with Noisy Labels*  
Paper: [https://arxiv.org/pdf/2006.13554v1.pdf](https://arxiv.org/pdf/2006.13554v1.pdf)

### Label Smoothing
Train using label smoothing

### CE-MAE
Train using CE at early stage and using MAE after

### Augmentation-filter
Filtering noise image using several augmentation