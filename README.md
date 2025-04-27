# NICU-Noise-Shield
# Real-time Active Noise Cancellation System

A Python-based project implementing a real-time active noise cancellation (ANC) system with the following features:

- **Dual-channel ANC** using the LMS algorithm  
- **Real-time audio level visualization** (dB)  
- **Periodic sound-source classification** (CNN-based inference)  
- **Trigger-based ANC suspension** when a specific noise source is detected  
- **Console display** of current noise source and confidence

---

## Features

1. **Dual‑channel noise cancellation**: Continuously acquire stereo input, apply LMS ANC, output cleaned stereo audio.  
2. **Real‑time level plot**: Compute RMS level of processed audio and display dynamic dB curve.  
3. **Sound‑source detection**: Record mono audio every `RECORD_INTERVAL` seconds, preprocess into Mel spectrogram, and classify via pretrained CNN.  
4. **Trigger control**: Automatically mute ANC output when the detected source matches `TRIGGER_SOURCE`.  
5. **Status display**: Console logs show `Source: <name> | Confidence: <value>` in real time.


---

## Installation

1. **Clone the repository**:
   ```bash
   git clone [https://your.repo.url/project.git](https://github.com/MiMickyyy/NICU-Noise-Shield.git)
   cd project
   ```

2. **Create a virtual environment** :
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install manually:
   ```bash
   pip install sounddevice numpy scipy matplotlib librosa opencv-python tensorflow
   ```

---

## Configuration

All parameters reside in **`config.py`**:

- **Audio I/O**: `SAMPLE_RATE`, `BLOCK_SIZE`, `INPUT_DEVICE`, `OUTPUT_DEVICE`  
- **ANC**: `ANC_CHANNELS`, `FILTER_LEN`, `STEP_SIZE`  
- **Sound Detection**: `RECORD_SECONDS`, `RECORD_INTERVAL`, `N_FFT`, `HOP_LENGTH`, `MODELS_DIR`, `CLASS_LABELS`, `CONF_THRESHOLD`, `NORMAL_CLASS_INDEX`  
- **Visualization**: `LEVEL_MAX_POINTS`, `LEVEL_VMIN`, `LEVEL_VMAX`  

Set **`TRIGGER_SOURCE`** in `config.py` to the label that should pause ANC output.

---

## Usage

### 1) Quick Start

```bash
python run.py
```  

This will:

- Start the audio I/O stream  
- Launch LMS ANC processing  
- Open a dynamic dB-level plot window  
- Begin background source detection  
- Mute ANC output when `TRIGGER_SOURCE` is detected

### 2) Using the Controller

Alternatively, run the integrated controller:

```bash
python -m controller
```  

---

## Module Descriptions

- **`audio_io.py`**: Wraps `sounddevice.Stream` for stereo capture/playback.  
- **`anc_lms.py`**: Implements adaptive LMS filters for each channel.  
- **`spectrogram_visualizer.py`**: Contains `LevelVisualizer` for RMS-level plotting.  
- **`source_detector.py`**: Records, preprocesses, and classifies incoming noise using a CNN.  
- **`controller.py`**: Orchestrates I/O, ANC, visualization, and detection in one class.  
- **`run.py`**: Simplified entry point that sets up modules and enters the main loop.  

---


```

---

## License



---

## Acknowledgments

- Based on the LMS algorithm and Mel-spectrogram classification workflows.
- Uses `sounddevice`, `librosa`, `tensorflow`, and `matplotlib`.

