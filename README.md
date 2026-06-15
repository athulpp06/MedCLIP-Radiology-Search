🔍 MedCLIP Radiology Semantic Search
A cross-modal, text-to-image semantic search engine capable of retrieving complex, unstructured radiology scans based purely on natural language clinical queries.

Unlike traditional classification models that predict from a fixed list of diseases, this application uses a Dual-Encoder Architecture (MedCLIP) and Contrastive Learning to project medical images and clinical text into a shared 512-dimensional vector space. This allows users to search for specific anatomies, modalities, and pathologies (e.g., "Axial CT scan of the chest showing a pleural effusion") and instantly retrieve the highest matching scans.

📊 Model Performance
The architecture was evaluated using Top-K Retrieval Accuracy (Recall@K) on a holdout set:

Recall@1: 73.44%

Recall@5: >99.00%

Recall@10: >99.00%

🧠 Architecture & Training Pipeline
This project utilizes two highly parameterized backbones:

Vision Encoder: ResNet (projects radiological image patches)

Text Encoder: Bio_ClinicalBERT (projects clinical sentence context via the [CLS] token)

The 15+3 Training Strategy:
The model was trained on Kaggle using a two-phase InfoNCE contrastive loss approach:

Phase 1 (Global Alignment): 15 epochs with frozen backbones (Learning Rate: 1e-4) to safely train the linear projection layers.

Phase 2 (Fine-Tuning): 3 epochs with unfrozen backbones (Learning Rate: 1e-5) to optimize deep internal visual and linguistic parameters, dropping the contrastive loss to a final 0.934.

🔗 View the full training loop and hyperparameter configuration here: Kaggle Training Notebook

📂 Dataset Disclaimer
This model was trained on the ROCO (Radiology Objects in COntext) Radiology Dataset.
Because the dataset is ~7.1 GB and contains ~81.8K high-resolution files, the dataset is NOT included in this GitHub repository. To run this application locally, you must download the dataset separately and map it to the local directory.

🚀 Local Setup & Installation Instructions
1. Hardware Requirements
For optimal performance during vector database generation and local inference, the following hardware configuration is recommended:

GPU: CUDA-compatible dedicated GPU (e.g., NVIDIA RTX 2000 Ada generation or equivalent)

RAM: 32GB System RAM recommended for handling large FAISS vector arrays

OS: Ubuntu Linux or Windows 11

2. Clone the Repository
Bash
git clone https://github.com/athulpp06/MedCLIP-Radiology-Search.git
cd MedCLIP-Radiology-Search
3. Install Dependencies
Ensure you have a Python virtual environment active, then install the required libraries:

Bash
pip install torch torchvision faiss-cpu pandas numpy pillow streamlit transformers tqdm
(Note: Use faiss-gpu instead of faiss-cpu if you wish to run the vector database entirely on VRAM).

4. Download the Data & Weights
Download the ROCO dataset and place the CSV files inside a data/ folder, and the images inside data/test_set/.

Download the trained expert weights (medclip_expert_v2.pth) from the Kaggle notebook outputs and place the file in the root directory of this repository.

5. Build the Vector Database
Before launching the app, you must map the medical images into the 512-dimensional vector space. Run the index builder to generate the FAISS index and metadata:

Bash
python build_index.py
This will generate vector_index.faiss and metadata.csv.

6. Launch the Application
Start the Streamlit search engine UI:

Bash
streamlit run app.py
🛠️ Repository Structure
app.py - The main Streamlit web interface and real-time inference pipeline.

build_index.py - The script responsible for projecting images through the Vision model and saving the FAISS vector database.

src/model.py - The PyTorch blueprint defining the Dual-Encoder MedCLIP architecture.

README.md - Project documentation.