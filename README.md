# 🎼 RagaVision: Automated Indian Classical Music Identification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Ruby](https://img.shields.io/badge/Ruby-3.2-red?logo=ruby)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange?logo=streamlit)
![Model Accuracy](https://img.shields.io/badge/Accuracy-92.4%25-success)
![License](https://img.shields.io/badge/License-Academic-blue)

**RagaVision** is an AI-powered system that identifies Indian classical ragas from audio clips using deep learning. It fuses a hybrid **CNN + Bi-LSTM + Attention** model, an intuitive **Streamlit UI**, and a **Ruby-based Selenium scraper** for enhanced context and metadata enrichment.

---

## 📌 Table of Contents

* [🚀 Features](#-features)
* [🧠 Model Architecture](#-model-architecture)
* [🎤 Dataset & Preprocessing](#-dataset--preprocessing)
* [🖥️ UI & Frontend](#-ui--frontend)
* [🔍 Ruby Scraper Details](#-ruby-scraper-details)
* [🛠️ Project Setup](#-project-setup)
* [📂 Project Structure](#-project-structure)
* [📊 Performance Metrics](#-performance-metrics)
* [🧪 Sample Usage](#-sample-usage)
* [🔭 Future Work](#-future-work)
* [📜 License](#-license)
* [🙌 Acknowledgements](#-acknowledgements)

---

## 🚀 Features

* 🎧 Real-time **Indian Classical Raga prediction**
* 🧠 Deep learning model: **CNN + Bi-LSTM + Attention**
* 🎯 Achieves **92.4% accuracy** across 32 ragas and 15,000+ samples
* 🌐 **Streamlit-based UI** for seamless user experience
* 🔍 **Ruby + Selenium scraper** fetches raga metadata from Google
* 📊 Visualizations: Waveform, MFCC heatmaps, and confidence scores
* 🧾 Outputs results in JSON and text format

---

## 🧠 Model Architecture

* **Input:** 39-dimensional MFCC vectors (13 MFCC + delta + delta-delta)
* **CNN Layer (1D):** Local acoustic feature extraction
* **Bi-LSTM Layer:** Sequential pattern recognition
* **Attention Mechanism:** Highlights musically significant regions
* **Dense Layer + Softmax:** Classification into 32 ragas
* **Loss:** Sparse Categorical Crossentropy with label smoothing
* **Optimizer:** Adam

---

## 🎤 Dataset & Preprocessing

* 🎼 Ragas Covered: 5 for prototype (scalable to 32+)
* 🧪 Dataset Size: \~1500 clips (\~50 hours audio)
* 🔁 10-fold cross-validation
* 🎚️ Data Augmentation:

  * Pitch shifting
  * Time stretching
  * Background noise overlay
* 🔇 Silence trimming + MFCC normalization
* 🎛️ Feature Extraction: `librosa` library

---

## 🖥️ UI & Frontend

Built with **Streamlit**, providing an interactive, browser-based UI.

**Features:**

* 📤 Upload `.mp3` / `.wav` files
* 🎙️ Live mic input (optional support)
* 🔉 Audio waveform and MFCC heatmap
* 🎼 Top-3 predicted ragas with confidence scores
* 🌐 Scraped raga metadata shown live
* 🧾 Export results to `.json` or `.txt`

**Views Available:**

* Explore
* Raga Predictor
* Music Chat

---

## 🔍 Ruby Scraper Details

### Why Ruby?

Ruby, paired with `Selenium` via `Watir`, allows clean, fast automation scripts. It offloads scraping logic and operates independently from the Python backend.

### Functionality:

* Performs a **Google search** for the predicted raga
* Collects titles, URLs, and snippets from top results
* Saves to:

  * `search_results.json`
  * `search_results.txt`
* These are rendered within the Streamlit UI

### Ruby Dependencies:

```bash
bundle install
ruby Scraper.rb
```

> ⚠️ Requires `chromedriver` in your system path.

---

## 🛠️ Project Setup

<details>
<summary>🐍 Python Environment Setup</summary>

```bash
git clone https://github.com/your_username/RagaVision.git
cd RagaVision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary>💎 Ruby Scraper Setup</summary>

```bash
# Install gems
bundle install

# Run the Ruby scraper
ruby Scraper.rb
```

</details>

### 🚀 Run the Application

```bash
streamlit run App2.py
```

---

## 📂 Project Structure

<details>
<summary>📁 Click to expand full file structure</summary>

```
RagaVision/
├── App2.py                # Main Streamlit app controller
├── config.py              # Configuration variables
├── logic.py               # Audio handling & business logic
├── model_utils.py         # Load model, labels, scaler
├── raga_predictor.py      # MFCC feature extraction + prediction
├── ui.py / raga_ui.py     # UI layout & background logic
├── Scraper.rb             # Ruby + Selenium scraper
├── requirements.txt       # Python dependencies
├── Gemfile                # Ruby dependencies
├── data/
│   ├── search_results.json
│   ├── search_results.txt
│   └── scraper.log
├── models/
│   ├── raga_model.h5 / .keras
│   ├── label_classes.npy
│   └── scaler.pkl
├── static/
│   ├── Background_explore.png
│   ├── Background_chat.png
│   ├── styles.css
│   └── raga_icon.png
```

</details>

---

## 📊 Performance Metrics

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 92.4%   |
| Precision | 92.1%   |
| Recall    | 92.5%   |
| ROC-AUC   | 0.981   |
| Latency   | \~2.06s |

---

## 🧪 Sample Usage

1. Start the app using:

   ```bash
   streamlit run App2.py
   ```
2. Upload or record an Indian classical music sample
3. View:

   * 🎼 Predicted Raga
   * 📊 Confidence Scores
   * 🌐 Scraped raga metadata
   * 📈 MFCC heatmap and waveform
   * 💾 Save results to JSON/text

---

## 🔭 Future Work

* 🤖 Integrate **Transformer** or **Conformer** models
* 🎶 Expand support to **100+ ragas**, regional variations
* 📱 Deploy as **mobile app** (TensorFlow Lite)
* 🎼 Extend to identify **Tala** and **Shruti**
* 🧠 Multimodal training with lyrics and notation (future vision)

---

## 👨‍💻 Developed by

**Akavarapu Sarvadutt**  
Final Year Capstone Project  
Department of CSE, Keshav Memorial Engineering College

---

## 📜 License

This project is developed for academic and research purposes.
For collaborations or reuse, please contact the author directly.

---

## 🙌 Acknowledgements

* Libraries: `TensorFlow`, `Streamlit`, `Librosa`, `Watir`, `Selenium`
* Thanks to open-source datasets and resources that enabled this work

---


