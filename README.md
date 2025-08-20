# 🍎🍌🍊 Fruit Image Classifier

## 🚀 Overview

This project is a **CNN-based image classifier** that can identify different types of fruits such as **Apple, Banana, and Orange**.  
It is built using **TensorFlow/Keras** and trained on the [Fruits360 Dataset](https://www.kaggle.com/datasets/moltean/fruits).

---

## 📂 Dataset

Due to the dataset size (~4GB), it is **not included** in this repository.  
You can download it from Kaggle:

👉 [Download Fruits360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)

After downloading, extract it like this:

```
fruit-classifier/
│── dataset/
│   ├── train/
│   ├── test/
│── src/
│── README.md
```

If you just want to **test the pipeline quickly**, a small `sample_dataset/` is provided.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Abhishek28Sharma/fruit-classifier.git
cd fruit-classifier
pip install -r requirements.txt
```

---

## 🏋️ Training

To train the model on your dataset, run:

```bash
python src/train.py
```

✔️ The trained model will be saved as **`fruit_classifier.h5`**  
✔️ A training accuracy plot will be saved as **`training_plot.png`**

**Example Training Output:**

```
Epoch 1/10
200/200 [==============================] - 45s 223ms/step - loss: 0.8921 - accuracy: 0.7130 - val_loss: 0.4123 - val_accuracy: 0.8725
...
Epoch 10/10
200/200 [==============================] - 40s 200ms/step - loss: 0.1102 - accuracy: 0.9654 - val_loss: 0.0856 - val_accuracy: 0.9783
```


## 🔮 Prediction

Once training is complete, you can test with a single image:

```bash
python src/predict.py --image dataset/test/apple/apple1.jpg
```

**Example Output:**

```
Predicted: apple
```

**Sample Test Image:**

<img src="sample_dataset/apple/apple1.jpg" alt="Apple Example" width="200">

---

## 📊 Results

- Achieves **85–95% accuracy** with a small dataset.
- Learns to classify fruits such as 🍎 Apple, 🍌 Banana, 🍊 Orange.

---

## 📌 Future Improvements

- Add more fruit categories
- Train on larger dataset
- Deploy as a **web app** using Streamlit/Flask
