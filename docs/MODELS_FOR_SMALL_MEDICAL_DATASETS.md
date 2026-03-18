# Best Models for Small Medical Image Datasets (Low Annotation)

## 🎯 **Criteria for Selection:**
- ✅ Pretrained on ImageNet (transfer learning)
- ✅ Can freeze backbone (prevent overfitting)
- ✅ Lightweight (fewer parameters = less overfitting risk)
- ✅ Fast training and inference
- ✅ Proven performance on medical imaging
- ✅ Works well with 100-1000 images

---

## 📊 **Model Comparison Table**

| Model | Parameters | Size (MB) | Speed | Medical Use | Freeze Support |
|-------|-----------|-----------|-------|-------------|----------------|
| SqueezeNet | 1.2M | ~5 | ⭐⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes |
| MobileNetV2 | 3.5M | ~14 | ⭐⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes |
| MobileNetV3 | 4.2M | ~17 | ⭐⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes |
| EfficientNet-B0 | 5.3M | ~21 | ⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes |
| ShuffleNetV2 | 2.3M | ~9 | ⭐⭐⭐⭐⭐ | ✅ Good | ✅ Yes |
| ResNet-18 | 11.7M | ~47 | ⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes |
| ResNet-34 | 21.8M | ~87 | ⭐⭐⭐ | ✅ Good | ✅ Yes |
| DenseNet-121 | 8.0M | ~32 | ⭐⭐⭐ | ✅ Good | ✅ Yes |
| VGG-16 | 138M | ~528 | ⭐⭐ | ⚠️ Too large | ✅ Yes |
| AlexNet | 61M | ~233 | ⭐⭐ | ⚠️ Too large | ✅ Yes |

---

## 🏆 **Top Recommendations for Medical Imaging**

### **1. SqueezeNet (Currently Using) ⭐⭐⭐⭐⭐**

**Why it's perfect:**
- ✅ Smallest model (1.2M parameters)
- ✅ Fastest training and inference
- Excellent for small datasets
- Excellent for medical imaging

**Specs:**
- Parameters: 1,235,496
- Size: ~5 MB
- ImageNet Accuracy: 58.1%
- Speed: Very Fast

**Best for:**
- Very small datasets (<500 images)
- Fast experimentation
- Resource-constrained environments
- Real-time inference needed

**Code:**
```python
import torchvision.models as models

model = models.squeezenet1_1(pretrained=True)
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False
```

---

### **2. MobileNetV2 ⭐⭐⭐⭐⭐**

**Why it's excellent:**
- ✅ Very lightweight (3.5M parameters)
- ✅ Depthwise separable convolutions (efficient)
- ✅ Excellent for mobile/edge devices
- ✅ Great for medical imaging
- ✅ Better accuracy than SqueezeNet

**Specs:**
- Parameters: 3,504,872
- Size: ~14 MB
- ImageNet Accuracy: 72.0%
- Speed: Very Fast

**Best for:**
- Small to medium datasets (200-1000 images)
- Better accuracy needed than SqueezeNet
- Mobile/edge deployment
- Medical imaging applications

**Code:**
```python
import torchvision.models as models

model = models.mobilenet_v2(pretrained=True)
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False
```

---

### **3. MobileNetV3 ⭐⭐⭐⭐⭐**

**Why it's excellent:**
- ✅ Latest MobileNet (improved over V2)
- ✅ AutoML-optimized architecture
- ✅ Better accuracy than V2
- ✅ Still lightweight (4.2M parameters)
- ✅ Excellent for medical imaging

**Specs:**
- Parameters: 4,231,976
- Size: ~17 MB
- ImageNet Accuracy: 74.0%
- Speed: Very Fast

**Best for:**
- Small to medium datasets (200-1000 images)
- Best MobileNet performance needed
- Medical imaging with better accuracy
- Modern architecture benefits

**Code:**
```python
import torchvision.models as models

model = models.mobilenet_v3_small(pretrained=True)  # or mobilenet_v3_large
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False
```

---

### **4. EfficientNet-B0 ⭐⭐⭐⭐**

**Why it's excellent:**
- ✅ State-of-the-art efficiency
- ✅ Compound scaling (depth, width, resolution)
- ✅ Excellent accuracy/size tradeoff
- ✅ Great for medical imaging
- ✅ Slightly larger but more accurate

**Specs:**
- Parameters: 5,288,548
- Size: ~21 MB
- ImageNet Accuracy: 77.1%
- Speed: Fast

**Best for:**
- Medium datasets (300-1500 images)
- Best accuracy needed in lightweight model
- Medical imaging with complex features
- When you can afford slightly more parameters

**Code:**
```python
import torchvision.models as models

model = models.efficientnet_b0(pretrained=True)
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False
```

---

### **5. ShuffleNetV2 ⭐⭐⭐⭐**

**Why it's good:**
- ✅ Very efficient architecture
- ✅ Channel shuffle operation
- ✅ Lightweight (2.3M parameters)
- ✅ Fast inference
- ✅ Good for medical imaging

**Specs:**
- Parameters: 2,278,604
- Size: ~9 MB
- ImageNet Accuracy: 69.4%
- Speed: Very Fast

**Best for:**
- Small datasets (<500 images)
- Very fast inference needed
- Resource-constrained environments
- Medical imaging applications

**Code:**
```python
import torchvision.models as models

model = models.shufflenet_v2_x1_0(pretrained=True)
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
# Then unfreeze classifier only
```

---

### **6. ResNet-18 ⭐⭐⭐⭐**

**Why it's good:**
- ✅ Classic, proven architecture
- ✅ Residual connections (easier training)
- ✅ Moderate size (11.7M parameters)
- ✅ Excellent for medical imaging
- ✅ Very popular in medical research

**Specs:**
- Parameters: 11,689,512
- Size: ~47 MB
- ImageNet Accuracy: 69.8%
- Speed: Fast

**Best for:**
- Medium datasets (400-2000 images)
- When you need proven architecture
- Medical imaging research
- Better feature extraction needed

**Code:**
```python
import torchvision.models as models

model = models.resnet18(pretrained=True)
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
# Unfreeze classifier
for param in model.fc.parameters():
    param.requires_grad = True
```

---

### **7. ResNet-34 ⭐⭐⭐**

**Why it's acceptable:**
- ✅ Deeper than ResNet-18
- ✅ Better feature extraction
- ✅ Larger (21.8M parameters)
- ✅ Good for medical imaging
- ⚠️ Risk of overfitting with very small datasets

**Specs:**
- Parameters: 21,797,672
- Size: ~87 MB
- ImageNet Accuracy: 73.3%
- Speed: Moderate

**Best for:**
- Medium to large datasets (500-3000 images)
- When ResNet-18 isn't enough
- Complex medical features
- More capacity needed

**Code:**
```python
import torchvision.models as models

model = models.resnet34(pretrained=True)
# Freeze backbone
for param in list(model.parameters())[:-2]:  # All except last 2 layers
    param.requires_grad = False
```

---

### **8. DenseNet-121 ⭐⭐⭐**

**Why it's good:**
- ✅ Dense connections (feature reuse)
- ✅ Parameter efficient
- ✅ Good for medical imaging
- ✅ Moderate size (8.0M parameters)

**Specs:**
- Parameters: 7,978,856
- Size: ~32 MB
- ImageNet Accuracy: 74.4%
- Speed: Moderate

**Best for:**
- Medium datasets (400-2000 images)
- When feature reuse is beneficial
- Medical imaging with dense features
- Good accuracy/size balance

**Code:**
```python
import torchvision.models as models

model = models.densenet121(pretrained=True)
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False
```

---

## 🚫 **Models to AVOID for Small Datasets**

### **VGG-16/19**
- ❌ Too large (138M+ parameters)
- ❌ High overfitting risk
- ❌ Slow training
- ❌ Not suitable for <1000 images

### **AlexNet**
- ❌ Old architecture
- ❌ Large (61M parameters)
- ❌ Outdated
- ❌ Not recommended

### **Large ResNets (50, 101, 152)**
- ❌ Too many parameters
- ❌ Overfitting risk with small data
- ❌ Only for large datasets (>2000 images)

### **EfficientNet-B1 to B7**
- ❌ Too large for small datasets
- ❌ Overfitting risk
- ⚠️ Only use if you have >1000 images

---

## 📋 **Quick Selection Guide**

### **For <300 images:**
1. **SqueezeNet** (best choice)
2. **ShuffleNetV2** (alternative)
3. **MobileNetV2** (if you need better accuracy)

### **For 300-500 images:**
1. **MobileNetV2** (best balance)
2. **MobileNetV3** (better accuracy)
3. **SqueezeNet** (if speed critical)
4. **EfficientNet-B0** (if accuracy critical)

### **For 500-1000 images:**
1. **EfficientNet-B0** (best overall)
2. **MobileNetV3** (faster alternative)
3. **ResNet-18** (proven architecture)
4. **DenseNet-121** (good features)

### **For 1000-2000 images:**
1. **ResNet-18/34** (proven)
2. **EfficientNet-B0/B1** (state-of-the-art)
3. **DenseNet-121** (feature reuse)

---

## 🔧 **How to Use These Models in Your Code**

### **Template for Any Model:**

```python
import torch
import torch.nn as nn
import torchvision.models as models

def create_medical_classifier(model_name='mobilenet_v2', 
                              num_classes=2, 
                              freeze_backbone=True,
                              pretrained=True):
    """
    Create a medical image classifier with frozen backbone
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze pretrained features
        pretrained: Use ImageNet pretrained weights
    """
    
    # Load pretrained model
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.last_channel, num_classes)
        )
        # Freeze backbone
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
    
    elif model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=pretrained)
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Freeze backbone
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        # Freeze backbone
        if freeze_backbone:
            for param in list(model.parameters())[:-2]:
                param.requires_grad = False
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        # Freeze backbone
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
    
    return model

# Usage
model = create_medical_classifier(
    model_name='mobilenet_v2',
    num_classes=2,
    freeze_backbone=True,
    pretrained=True
)
```

---

## 📊 **Performance Comparison (Estimated)**

### **For ~400 Image Medical Dataset:**

| Model | Expected Accuracy | Training Time | Inference Speed |
|-------|------------------|---------------|-----------------|
| SqueezeNet | 85-90% | Fastest | Fastest |
| MobileNetV2 | 87-92% | Fast | Very Fast |
| MobileNetV3 | 88-93% | Fast | Very Fast |
| EfficientNet-B0 | 89-94% | Moderate | Fast |
| ResNet-18 | 88-93% | Moderate | Fast |
| ShuffleNetV2 | 84-89% | Fastest | Fastest |

*Note: Actual results depend on your specific dataset and task*

---

## 🎯 **Recommendations by Use Case**

### **1. Fastest Training Needed:**
- **SqueezeNet** or **ShuffleNetV2**

### **2. Best Accuracy Needed:**
- **EfficientNet-B0** or **MobileNetV3**

### **3. Best Balance:**
- **MobileNetV2** or **MobileNetV3**

### **4. Proven in Medical Research:**
- **ResNet-18** or **DenseNet-121**

### **5. Mobile/Edge Deployment:**
- **MobileNetV3** or **MobileNetV2**

### **6. Very Small Dataset (<300 images):**
- **SqueezeNet** (your current choice is perfect!)

---

## 🔄 **How to Switch Models in Your Code**

### **In `model.py`:**

```python
def create_medical_model(model_name='squeezenet', 
                        pretrained=True, 
                        num_classes=2,
                        freeze_backbone=True):
    """
    Create medical image classifier
    
    Args:
        model_name: 'squeezenet', 'mobilenet_v2', 'mobilenet_v3', 
                   'efficientnet_b0', 'resnet18', etc.
    """
    if model_name == 'squeezenet':
        # Your current implementation
        model = models.squeezenet1_1(pretrained=pretrained)
        model.classifier = nn.Sequential(...)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.last_channel, num_classes)
        )
        
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(576, num_classes)
        )
        
    # Freeze backbone if requested
    if freeze_backbone:
        if hasattr(model, 'features'):
            for param in model.features.parameters():
                param.requires_grad = False
        elif hasattr(model, 'conv1'):
            # For ResNet-like models
            for param in list(model.parameters())[:-2]:
                param.requires_grad = False
    
    return model
```

### **In `train_kfold.py` config:**

```python
config = {
    'model_name': 'mobilenet_v2',  # Change here!
    'freeze_backbone': True,
    # ... rest of config
}
```

---

## 📚 **References & Further Reading**

### **Papers:**
1. **SqueezeNet**: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
2. **MobileNetV2**: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
3. **MobileNetV3**: "Searching for MobileNetV3"
4. **EfficientNet**: "EfficientNet: Rethinking Model Scaling for CNNs"
5. **ResNet**: "Deep Residual Learning for Image Recognition"

### **Medical Imaging Applications:**
- All these models have been successfully used in:
  - Histopathology
  - Radiology
  - Dermatology
  - Ophthalmology
  - Pathology

---

## ✅ **Summary**

**Best Models for Small Medical Datasets (Ranked):**

1. ⭐⭐⭐⭐⭐ **SqueezeNet** - Smallest, fastest (your current choice!)
2. ⭐⭐⭐⭐⭐ **MobileNetV2** - Best balance
3. ⭐⭐⭐⭐⭐ **MobileNetV3** - Latest, best MobileNet
4. ⭐⭐⭐⭐ **EfficientNet-B0** - Best accuracy
5. ⭐⭐⭐⭐ **ResNet-18** - Proven in medical research
6. ⭐⭐⭐⭐ **ShuffleNetV2** - Very efficient
7. ⭐⭐⭐ **DenseNet-121** - Good feature reuse

**All support:**
- ✅ ImageNet pretrained weights
- ✅ Backbone freezing
- ✅ Fine-tuning
- ✅ Medical imaging applications

**Your current SqueezeNet choice is excellent for ~400 images!** 🎉

