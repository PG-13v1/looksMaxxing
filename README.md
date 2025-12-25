LooksMaxxing

LooksMaxxing is an AI-driven project that analyzes and scores facial attractiveness using landmark detection, symmetry metrics, and golden ratio indicators. Built with modern computer vision tools, this system provides interpretable feedback on facial features and beauty metrics.

🔗 GitHub Repository: https://github.com/PG-13v1/looksMaxxing.git

🧠 About

LooksMaxxing uses facial landmark extraction and aesthetic ratios to assess facial attractiveness based on established metrics like symmetry, proportion, and alignment. The project aggregates visual features to compute an overall attractiveness score and provides region-wise insights (e.g., eyes, nose, jawline) for personalized feedback.

🚀 Features

✔️ Facial landmark detection (e.g., 68+ keypoints)
✔️ Symmetry analysis between left/right facial regions
✔️ Golden ratio face proportion evaluation
✔️ Region-wise attractiveness scoring
✔️ Visual overlays for keypoints and measurements

🗂 Repository Structure
LooksMaxxing/
├── datasets/                   # Sample images for evaluation
├── models/                     # Pre-trained landmark models
├── src/
│   ├── face_detection.py       # Face detector logic
│   ├── landmark_extractor.py   # Facial landmark extraction
│   ├── metrics.py              # Symmetry & proportion calculations
│   ├── attractiveness.py       # Scoring logic
│   └── utils.py                # Helper functions
├── results/                    # Output visuals & reports
├── requirements.txt
├── main.py                     # Entry point script
└── README.md

🛠 Tech Stack

Python 3.x

OpenCV — Computer vision

Dlib / MediaPipe — Facial landmark extraction

NumPy / SciPy — Numeric computations

Matplotlib / Plotly — Visualizations

📦 Installation

Clone the repository

git clone https://github.com/PG-13v1/looksMaxxing.git
cd looksMaxxing


Create & activate virtual environment

python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows


Install dependencies

pip install -r requirements.txt

📊 Usage
🚀 Run the Main Script
python main.py --image path/to/photo.jpg

🎯 Example Output

The system generates:

A visual image with keypoints & overlays

Symmetry scores between facial halves

Golden ratio measurements

A final attractiveness score

🧩 Configuration

Modify configuration values (e.g., thresholds, model paths) in config.json or in the relevant Python modules to customize scoring and evaluation behavior.

📈 Goals & Roadmap

Future improvements include:

📌 Training custom landmark models with deep learning

📌 Adding real-time webcam support

📌 Voice & expression integration

📌 UI / web interface for easier interaction

📫 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to submit a pull request or open an issue to start the conversation.

📄 License

Include an open-source license like MIT or Apache 2.0 to clarify usage rights, if not already present.
