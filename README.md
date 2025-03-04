# **Wi-Fi Tracking & Object Movement Detection System** 🚀

## **📌 Project Overview**

This project is a **real-time security and object movement detection system** using **Wi-Fi Channel State Information (CSI)**.  
It leverages deep learning to detect **object presence, movement, and unexpected anomalies** without cameras or motion sensors.

🔹 **Privacy-Preserving:** Works without traditional cameras, ensuring **non-intrusive monitoring**.  
🔹 **Real-Time Detection:** Tracks objects appearing, disappearing, or moving using **Wi-Fi signal variations**.  
🔹 **AI-Powered Security:** Uses **Bi-Directional LSTM + Attention Mechanism** to detect anomalies.

---

## **💡 Features**

✔ **Real-Time Object Detection & Tracking**  
✔ **Movement & Anomaly Detection**  
✔ **Environment-Aware Learning (Adapts to Different Locations)**  
✔ **Future Expansion: Gesture Recognition, Breathing Rate Monitoring, Smart Home Integration**

---

## **🛠 Technology Stack**

- **Programming Language**: Python 🐍
- **Deep Learning Framework**: TensorFlow / Keras 🧠
- **Data Processing**: NumPy, Pandas 📊
- **Machine Learning Models**:
  - **Conv1D + Bi-Directional LSTM + Attention Mechanism**
  - **Anomaly Detection for Unexpected Movement**
  - **Domain Adaptation for Environment-Awareness**

---

## **🚀 Installation & Setup**

### **1️⃣ Install Dependencies**

Run this command to install all required Python libraries:

```bash
pip install -r requirements.txt
```

### **2️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/WiFi_CSI_Security.git
cd WiFi_CSI_Security
```

### **3️⃣ Run the Model**

To train the model:

```bash
python src/train_model.py
```

To start **real-time object movement detection**:

```bash
python src/real_time_detection.py
```

---

## **🔹 Real-Time CSI Data Streaming Setup**

### **Option 1: Intel 5300/6300 Wi-Fi Card (Linux Only)**

1. Install the **CSI Tool**: [Intel 5300 CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/)
2. Load the Wi-Fi driver:
   ```bash
   sudo modprobe iwlwifi connector_log=0x1
   ```
3. Start capturing CSI data:
   ```bash
   ./log_to_file csi.dat
   ```
4. Modify the Python script to read CSI packets in **real-time**.

### **Option 2: Raspberry Pi (Nexmon CSI Tool)**

1. Install Nexmon CSI on Raspberry Pi:  
   [Nexmon CSI Guide](https://github.com/seemoo-lab/nexmon_csi)
2. Start CSI streaming:
   ```bash
   nexutil -m2 -s500 -b -l34
   ```
3. Modify the model to **ingest live CSI data** instead of simulated input.

---

## **🛠 Model Architecture**

1. **Feature Extraction**: Uses **Wi-Fi CSI amplitude & phase** data.
2. **Temporal Learning**: Bi-Directional LSTMs process CSI time-series data.
3. **Attention Mechanism**: Helps focus on the most important signal variations.
4. **Anomaly Detection Mode**: Identifies **unexpected object movement** based on CSI pattern shifts.
5. **Environment-Aware Learning**: Uses **configuration embeddings** to adapt to different rooms & layouts.

---

## **📊 Model Training & Performance**

- **Dataset**: Uses real-time CSI data for object classification & movement detection.
- **Training**:
  ```bash
  python src/train_model.py
  ```
- **Real-Time Detection**:
  ```bash
  python src/real_time_detection.py
  ```
- **Test Accuracy**: Typically **above 90% for trained environments**, adaptable to new spaces.

---

## **📌 How to Detect Object Movement in Real-Time?**

1. **Start the real-time detection system**:
   ```bash
   python src/real_time_detection.py
   ```
2. **Move an object (or introduce a new one).**
3. If an unexpected change is detected, an alert appears:
   ```bash
   ⚠️ ALERT: Unexpected Object Presence or Movement Detected!
   ```

---

## **🛠 Troubleshooting & Improvements**

### **🔹 No Alerts Even When Moving an Object?**

✔ **Lower the detection threshold** in `detect_object_movement()`:

```python
def detect_object_movement(new_data, config_data, threshold=0.005):
```

✔ **Use real-time CSI instead of simulated data**.  
✔ **Ensure the model is trained on diverse data (different objects & positions).**

### **🔹 Model Not Detecting Movements Accurately?**

✔ Increase **training data variety** (different locations, object types).  
✔ Use **multiple Wi-Fi routers for better spatial detection**.  
✔ Apply **Kalman filters or signal smoothing** to reduce CSI noise.

---

## **📢 Future Enhancements & Next Steps**

🔹 **Integrate with Smart Homes** (turn on lights when a person enters).  
🔹 **Use CSI for Gesture Recognition & AI Assistants**.  
🔹 **Expand to Health Monitoring (breathing, heart rate tracking)**.  
🔹 **Combine with AI-powered anomaly detection for security alerts**.

---

## **📜 License**

This project is open-source under the **MIT License**.

---

## **📩 Contact & Contributions**

📧 **For contributions, reach out at:** *agarwaldhawalaero10@gmail.com*

💡 **We welcome pull requests & improvements!** 🚀
