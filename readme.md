# Pneumonia Detection System (Deep Learning + FastAPI)

This project is an **end-to-end Pneumonia Detection system** built using **Deep Learning (PyTorch)** for model development and **FastAPI** for model serving. It is designed both as a **deployable ML application** and as a **strong interview-ready project**, covering training, inference, API serving, and explainability.

---

## 1. Project Motivation (Interview Explanation)

Pneumonia is a life‑threatening respiratory infection where **early diagnosis from chest X‑ray images** is critical. Manual diagnosis is time‑consuming and subject to inter‑observer variability.

**Goal:**
- Build a reliable AI system that can **classify chest X‑rays as Pneumonia or Normal**
- Deploy the trained model as a **real‑time API service** usable by external applications

---

## 2. High‑Level Architecture

```
Chest X‑ray Image
        ↓
Image Pre‑processing
        ↓
Deep Learning Model (ResNet‑152)
        ↓
Softmax Probabilities
        ↓
FastAPI Backend
        ↓
JSON Response (Prediction + Confidence)
```

This separation ensures:
- Clean ML ↔ Backend boundaries
- Production‑readiness
- Easy model upgrades

---

## 3. Model Choice & Justification

### 🔹 Why ResNet‑152?
- Very deep CNN capable of learning **complex lung texture patterns**
- Residual connections prevent vanishing gradients
- Strong performance in medical image classification
- Pretrained on ImageNet → faster convergence

### Model Design
- Backbone: **ResNet‑152 (frozen)**
- Custom fully‑connected layer for pneumonia classification
- Transfer learning strategy to avoid overfitting

```text
ResNet‑152 Backbone (Frozen)
→ Global Average Pooling
→ Fully Connected Layer
→ Pneumonia / Normal
```

---

## 4. Dataset Pre‑processing Pipeline

Pre‑processing is critical in medical imaging to avoid bias and noise.

### Image‑Level Processing
- Resize to **224 × 224** (ImageNet standard)
- Convert grayscale → RGB (3‑channel)
- Center crop
- Normalize using ImageNet mean & std

```python
Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
```

### Why This Matters (Interview Point)
- Ensures compatibility with pretrained models
- Improves numerical stability
- Prevents shortcut learning from artifacts

---

## 5. Training Strategy

### Transfer Learning
- Backbone frozen
- Only classifier head trained

### Loss Function
- **CrossEntropyLoss** (multi‑class ready)

### Optimizer
- **Adam** with weight decay

### Validation
- Tracks:
  - Training loss
  - Validation loss
  - Validation accuracy

---

## 6. Model Training Flow

```text
Load Dataset
→ Apply Transforms
→ Forward Pass
→ Compute Loss
→ Backpropagation
→ Validation
→ Save Weights
```

This is implemented inside a reusable `fit()` method for clean experimentation.

---

## 7. Model Serving with FastAPI

The trained model is exposed via **FastAPI**, allowing real‑time inference.

### Why FastAPI?
- Asynchronous & fast
- Automatic Swagger UI
- Easy ML integration
- Production‑ready

### Server Startup

```bash
python3 server.py
```

Server runs at:
```
http://127.0.0.1:8000
```

Swagger Docs:
```
http://127.0.0.1:8000/docs
```

---

## 8. Prediction Pipeline (Runtime)

```text
Client Image Upload
→ PIL Image Loading
→ Preprocessing
→ Model Inference (No Grad)
→ Softmax Probabilities
→ JSON Response
```

### Sample API Response

```json
{
  "predicted_class": "PNEUMONIA",
  "best_prob": 0.982134,
  "probs": [0.0178, 0.9821]
}
```

---

## 9. Code Structure (Explained)

```text
pneumonia/
│── model/
│   ├── pneumonia_model.py   # CNN architecture + training
│   ├── pneumonia_predictor.py  # Inference logic
│
│── api/
│   ├── server.py            # FastAPI entry point
│
│── config/
│   ├── pneumonia_cfg.py     # Constants & labels
│
│── utils/
│   ├── logger.py            # Logging system
│
│── weights/
│   ├── model.pth            # Trained model weights
```

This modular layout is **industry‑standard** and interview‑friendly.

---

## 10. Logging & Monitoring

- Centralized logging for:
  - Model load
  - Predictions
  - Errors
- Helps in debugging and auditing predictions

---

## 11. Key Interview Talking Points

### Technical
- Transfer learning
- CNN feature extraction
- Softmax confidence interpretation
- API‑based ML deployment

### ML Engineering
- Model freezing
- Data normalization
- Memory cleanup (`torch.cuda.empty_cache()`)
- Inference vs training separation

### System Design
- Stateless prediction API
- Model versioning ready
- Scalable backend

---

## 12. Future Improvements (Strong Interview Add‑Ons)

- Grad‑CAM explainability
- Lung segmentation before classification
- Transformer‑based hybrid model
- Docker + cloud deployment
- CI/CD for model updates

---

## 13. Final Summary

This project demonstrates:
- Real‑world **medical AI application**
- Full **ML lifecycle understanding**
- **Backend + ML integration** skills
- Production‑oriented thinking

It is suitable for:
- Technical interviews
- ML engineer roles
- Research extensions
- Portfolio showcase

---

**Author:** Shriverdhan Pathak

