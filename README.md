# Novel-Molecules-using-XGBoost

🧪 NovelMol: Generating and Screening Novel Molecules with AI
Welcome to NovelMol, an AI-driven pipeline for de novo drug design. This project leverages MolGAN to generate chemically valid molecules and integrates machine learning tools like Chemprop to evaluate their drug-likeness and bioactivity.

🚀 Project Highlights
Molecule Generation using generative adversarial networks (MolGAN)

Drug-likeness Screening using Chemprop and other scoring models

Multiprocessing-enabled Workflow for high-throughput molecule evaluation

SMILES Support with RDKit validation and descriptor calculation

Fully modular and customizable for new targets or properties

🧬 Why NovelMol?
The discovery of new, drug-like molecules is critical yet costly. NovelMol aims to accelerate early-stage drug discovery using deep generative models and predictive machine learning, with applications in:

Cancer and neurodegenerative disease therapeutics

Target-specific inhibitor generation (e.g., TDP-43 inhibitors)

Cheminformatics and bioactivity prediction research

📦 Pipeline Overview
mermaid
Copy
Edit
graph TD
    A[MolGAN] --> B[Generated SMILES]
    B --> C[RDKit Validation]
    C --> D[Descriptor Calculation]
    D --> E[Chemprop Scoring]
    E --> F[Filtered Molecule Library]
Each component is modular and configurable via YAML or Python scripts.

🛠️ Technologies Used
MolGAN (Molecular GAN model)

RDKit for molecule validation & descriptors

Chemprop for property prediction

Python multiprocessing for scalable screening

Pandas, NumPy, tqdm for data handling and tracking

📁 Folder Structure
bash
Copy
Edit
NovelMol/
│
├── molgan/                # MolGAN implementation & pretrained weights
├── screening/             # Chemprop & descriptor filtering
├── data/                  # Input/output SMILES and logs
├── utils/                 # Helper scripts: validation, plotting
├── results/               # Output files: CSVs, visuals, top hits
├── config.yaml            # Configurations for training/screening
└── main.py                # Pipeline entry point
⚙️ How to Use
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/NovelMol.git
cd NovelMol
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the pipeline
bash
Copy
Edit
python main.py --config config.yaml
4. Output
filtered_molecules.csv: Molecules passing thresholds

descriptor_plots/: Drug-likeness and pIC50 score visuals

🧠 Sample Use Case
Design novel inhibitors for TDP-43, a key target in neurodegeneration and cancer:

yaml
Copy
Edit
target: TDP-43
scoring_model: chemprop_pretrained_model.pt
filters:
  - QED > 0.5
  - SA_score < 4.5
  - pIC50 > 6.0
📊 Visualization Example

✅ Roadmap
 Basic MolGAN integration

 Chemprop-based screening

 Reinforcement learning for target-specific generation

 GUI or Streamlit-based web dashboard

 Integration with molecular docking (AutoDock or GNINA)

🤝 Contributing
Pull requests are welcome! If you have ideas for new filters, scoring models, or visualizations, feel free to fork the repo and contribute.

📄 License
This project is licensed under the MIT License.

👤 Author
Aarya Vashishth
Bioinformatics | Drug Discovery | AI in Biomedicine
LinkedIn | Portfolio

Would you like me to generate a visual logo/banner or add a badge section (e.g., Python version, license, stars)?








You said:
yes


ChatGPT said:
Great! Here's an enhanced version of your README.md with:

🔖 Badges (license, Python version, stars, etc.)

🎨 Banner logo

🧪 Molecular emoji accents

🌟 A polished developer-friendly look

markdown
Copy
Edit
<p align="center">
  <img src="https://your-banner-image-link.com" alt="NovelMol Banner" width="100%" />
</p>

<h1 align="center">🧪 NovelMol</h1>
<p align="center">AI-Powered Pipeline for Generating and Screening Novel Drug-Like Molecules</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/github/license/yourusername/NovelMol">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8%2B-blue">
  <img alt="Issues" src="https://img.shields.io/github/issues/yourusername/NovelMol">
  <img alt="Stars" src="https://img.shields.io/github/stars/yourusername/NovelMol?style=social">
</p>

---

## 🌟 Overview

**NovelMol** is an open-source, modular pipeline for *de novo* drug discovery. It combines:
- 🧬 **MolGAN** to generate new molecules as SMILES
- 🔬 **RDKit** to ensure chemical validity
- 🧠 **Chemprop** and ML filters to score drug-likeness and bioactivity
- ⚙️ Multiprocessing for efficient high-throughput screening

Ideal for **biotech researchers**, **AI drug discovery** enthusiasts, and **computational biologists** working on novel targets like **TDP-43** in neurodegeneration and cancer.

---

## 📦 Key Features

✅ MolGAN-based SMILES generation  
✅ RDKit-based validation & descriptor calculation  
✅ Chemprop scoring and threshold-based filtering  
✅ Clean CSV output and descriptor visualizations  
✅ Multiprocessing-enabled pipeline for speed  

---

## 🧪 Technologies

- **MolGAN**: Deep generative model for molecules  
- **Chemprop**: Message passing neural network (MPNN)  
- **RDKit**: Cheminformatics toolkit  
- **Python 3.8+**, **tqdm**, **NumPy**, **pandas**

---

## 📁 Repository Structure

NovelMol/
├── molgan/ # Generator module
├── screening/ # Filters and ML scoring
├── utils/ # Helpers (validation, metrics)
├── data/ # Input/output files
├── results/ # Molecule results & visualizations
├── config.yaml # Config file
└── main.py # Pipeline entry point

yaml
Copy
Edit

---

## ⚙️ Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/NovelMol.git
cd NovelMol
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the pipeline
bash
Copy
Edit
python main.py --config config.yaml
📊 Output
filtered_molecules.csv: Molecules passing thresholds

results/: Plots of QED, SA Score, pIC50, and descriptor distributions

logs/: Screening progress and rejected molecules

💡 Sample Config
yaml
Copy
Edit
target: TDP-43
scoring_model: chemprop_model.pt
filters:
  QED: ">=0.6"
  SA_score: "<=4.5"
  pIC50: ">=6.5"
🛣️ Roadmap
 Baseline: MolGAN + Chemprop integration

 Reward optimization via RL

 Target docking (AutoDock/Gnina integration)

 Jupyter interface & Streamlit dashboard

 AWS/GPU-friendly parallel deployment

🤝 Contributing
We welcome PRs and collaborations from computational chemists, ML researchers, and bioinformaticians! Open an issue or fork this project to suggest enhancements.

📄 License
Distributed under the MIT License. See LICENSE for details.

👤 Author
Aarya Vashishth
Bioinformatics | AI Drug Discovery | Cancer & Neurodegeneration Research
🔗 LinkedIn
📫 Email: aarya.vashishth@email.com

