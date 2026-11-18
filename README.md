# AI-Powered Emotion Prediction Reliability Analysis

An end-to-end machine learning pipeline designed to evaluate the reliability and predict the misclassification risks of a custom-trained **YOLOv8** emotion recognition model.

## 🧠 Project Overview
While primary deep learning models (such as YOLOv8) can accurately classify facial expressions into distinct emotion categories, they are still prone to errors caused by lighting, blur, contrast, or ambiguous facial features. 

This project introduces a secondary meta-model (**Decision Tree Classifier**) that takes image quality metrics and primary model predictions as inputs to predict whether the primary YOLOv8 model's emotion prediction is likely to be incorrect (**Error Risk Prediction**).

---

## 🚀 Key Features & Workflow
1. **Primary Emotion Detection:** Utilizes a fine-tuned YOLOv8 nano model (`yolov8n-emo.pt`) to detect faces and classify them into 8 basic emotion categories (*anger, contempt, disgust, fear, happiness, neutrality, sadness, surprise*).
2. **Image Quality Feature Extraction:** Extracts low-level visual and geometric features from test images:
   * **Brightness & Contrast:** Mean and standard deviation of grayscale pixel intensities.
   * **Blur Detection:** Laplacian variance to measure image sharpness.
   * **Face Size:** The ratio of the detected bounding box area to the total image area.
   * **Model Confidence:** Primary YOLOv8 confidence scores.
3. **Error Risk Modeling:** Employs a Scikit-Learn `Pipeline` with a `DecisionTree` classifier to evaluate failure patterns and flag high-risk predictions.
4. **Rigorous Validation:** Evaluated using Stratified 5-Fold Cross-Validation, confusion matrices, and detailed performance metrics (Accuracy, Precision, Recall, F1-Score).

---

## 📊 Dataset & Classes
* **Dataset Structure:** Curated from multiple emotion datasets containing test images and bounding box coordinate annotations.
* **Emotion Classes (8):**
  - `0`: Anger
  - `1`: Contempt
  - `2`: Disgust
  - `3`: Fear
  - `4`: Happiness
  - `5`: Neutrality
  - `6`: Sadness
  - `7`: Surprise

---

## 🛠️ Tech Stack & Libraries
* **Python** 
* **Ultralytics YOLOv8** (Computer Vision & Object Detection)
* **Scikit-Learn** (Decision Tree Classifier, Pipelines, Cross-Validation, Metrics)
* **OpenCV & NumPy** (Image processing & feature extraction)
* **Pandas & Seaborn / Matplotlib** (Data analysis & visualization)

---

## ⚙️ Usage / Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/sedaygit/Yapay_Zeka_Destekli_Duygu_Tahmin_Guvenilirligi_Analizi.git](https://github.com/sedaygit/Yapay_Zeka_Destekli_Duygu_Tahmin_Guvenilirligi_Analizi.git)
   '''


2. Environment Setup: Open the notebook (AI_Powered_Emotion_Prediction_Reliability_Analysis.ipynb) in Google Colab or a local Jupyter environment.

## ⚙️ Usage / Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/sedaygit/(AI_Powered_Emotion_Prediction_Reliability_Analysis.git
   ```

2. Open the Jupyter Notebook or run the script in a Google Colab environment.

3. Ensure your dataset (`emotions56.zip`) and YOLOv8 weights (`yolov8n-emo.pt`) are correctly linked.

4. Run the pipeline cells to extract features, train the error-risk Decision Tree model, and generate evaluation plots.

## 👤 Author

**Seda Yalçın**

- GitHub: [@sedaygit](https://github.com/sedaygit)
- LinkedIn: [Seda Yalçın]([https://linkedin.com](https://www.linkedin.com/in/seda-yalcin/))

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
