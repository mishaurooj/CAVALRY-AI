# CAVALRY-AI

## Cybersecurity Agentic Vision-Action Large Language Model with Risk-aware hYbrid Intelligence

CAVALRY-AI is an agent-driven cybersecurity intelligence framework designed for binary intrusion detection and multiclass attack classification using the CSE-CIC-IDS2018 benchmark dataset. The framework combines flow-level network analytics, Cyber Memory Graph (CMG) contextual reasoning, Risk-Weighted Agent Routing (RWAR), and a Multi-Agent Intelligence Layer (MAIL) to provide accurate threat detection, contextual interpretation, and security response recommendations.

---

## Repository Structure

```text
CAVALRY-AI/
│
├── Code/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_binary.py
│   ├── train_multiclass.py
│   ├── mail_ablation.py
│   ├── generate_figures.py
│   └── requirements.txt
│
├── Results/
│   ├── Binary/
│   ├── Multiclass/
│   ├── MAIL_Ablation/
│   └── Figures/
│
├── Trained-models/
│   ├── Binary/
│   └── Multiclass/
│
├── CAVALRY_AI_Architecture.png
├── README.md
└── LICENSE
```

---

# Proposed CAVALRY-AI Architecture

The framework consists of six sequential processing layers.

| Layer | Description |
|---------|-------------|
| IAL | Input Acquisition Layer |
| FEL | Feature Engineering Layer |
| CMGL | Cyber Memory Graph Layer |
| RWARL | Risk-Weighted Agent Routing Layer |
| MAIL | Multi-Agent Intelligence Layer |
| TCRL | Threat Classification and Response Layer |

---

## Input Acquisition Layer (IAL)

The IAL collects network telemetry and cybersecurity evidence.

### Inputs

* Network flow records
* IDS alerts
* Threat intelligence feeds
* Vulnerability databases
* Security artifacts

---

## Feature Engineering Layer (FEL)

The FEL transforms raw traffic into structured intelligence.

### Feature Categories

| Category | Description |
|-----------|-------------|
| Statistical Features | Packet and flow statistics |
| Temporal Features | Timing behavior |
| Protocol Features | Transport protocol characteristics |
| Port Features | Service-port relationships |
| RWAR Features | Risk-aware indicators |
| CMG Features | Graph frequency and degree metrics |

---

## Cyber Memory Graph Layer (CMGL)

CMGL models cybersecurity entities and their relationships.

### Entities

* Hosts
* IP addresses
* Ports
* Services
* Alerts
* Threat indicators

### Relationships

* Communication
* Interaction
* Temporal correlation
* Service association

---

## Risk-Weighted Agent Routing Layer (RWARL)

RWAR dynamically activates cybersecurity agents according to estimated threat severity.

### Low Risk Route

Activated Agents:

* Sentinel Agent (SA)
* Hunter Agent (HA)
* Oracle Agent (OA)

### Medium Risk Route

Activated Agents:

* SA
* HA
* OA
* GraphMind Agent (GMA)
* Raptor Agent (RA)

### High Risk Route

Activated Agents:

* SA
* HA
* OA
* Specter Agent (SPA)
* GMA
* RA
* ForensiX Agent (FA)
* Shield Agent (SHA)

---

## Multi-Agent Intelligence Layer (MAIL)

MAIL performs collaborative cybersecurity analysis.

| Agent | Responsibility |
|---------|---------------|
| SA | Risk assessment and alert prioritization |
| HA | Intrusion detection and anomaly analysis |
| OA | Threat intelligence correlation |
| SPA | Malware behavior profiling |
| GMA | Context retrieval from CMG |
| RA | Adaptive retrieval and evidence fusion |
| FA | Root-cause analysis and forensic reasoning |
| SHA | Response planning and mitigation generation |
| CA | Decision orchestration and confidence aggregation |

---

## Threat Classification and Response Layer (TCRL)

TCRL evaluates binary intrusion detection and multiclass attack classification performance.

### Binary Detection

Classes:

* Benign
* Attack

### Multiclass Detection

Classes:

* Brute Force
* DDoS
* Botnet
* Web Attack
* Infiltration
* Port Scan
* U2R
* R2L
* Benign

---

# Dataset

## CSE-CIC-IDS2018

This study uses the CSE-CIC-IDS2018 benchmark intrusion detection dataset.

Dataset Source:

https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv

Original Provider:

Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)

### Dataset Characteristics

| Property | Value |
|-----------|---------|
| Dataset | CSE-CIC-IDS2018 |
| Source | University of New Brunswick |
| Traffic Type | Realistic Enterprise Traffic |
| Attacks | Multiple Attack Families |
| Features | 80 Network Flow Features |
| Labels | Binary and Multiclass |

### Important Attributes

* Destination Port
* Protocol
* Flow Duration
* Total Forward Packets
* Total Backward Packets
* Label

---

# Experimental Feature Configurations

| Configuration | Description |
|--------------|-------------|
| A0 | Core Statistical Flows |
| A1 | Core + Port + Protocol Context |
| A2 | Temporal Behavioral Extension |
| A3 | Full Numeric Representation |
| A4 | Numeric + RWAR Intelligence |
| A5 | Numeric + CMG Intelligence |
| A6 | Full CAVALRY-AI Architecture |
| A6b | Optimized Feature Core |
| A7 | TCP Diagnostic Baseline |

---

# Decision Engines

To evaluate feature robustness independently of a single learning strategy, five decision engines are employed.

| Engine | Abbreviation |
|----------|-------------|
| Context Forest Engine | CFE |
| Adaptive Boost Engine | ABE |
| Ensemble Decision Engine | EDE |
| Linear Threat Engine | LTE |
| Probabilistic Security Engine | PSE |

---

# Environment Setup

## Step 1: Clone Repository

```bash
git clone https://github.com/mishaurooj/CAVALRY-AI.git

cd CAVALRY-AI
```

---

## Step 2: Create Conda Environment

```bash
conda create -n cavalry-ai python=3.10 -y
```

---

## Step 3: Activate Environment

### Windows

```bash
conda activate cavalry-ai
```

### Linux / macOS

```bash
source activate cavalry-ai
```

---

## Step 4: Install Dependencies

```bash
pip install -r Code/requirements.txt
```

---

# Running the Framework

## Binary Intrusion Detection

```bash
python Code/train_binary.py
```

---

## Multiclass Attack Classification

```bash
python Code/train_multiclass.py
```

---

## MAIL Agent Ablation Study

```bash
python Code/mail_ablation.py
```

---

## Generate Figures

```bash
python Code/generate_figures.py
```

---

# Reproducing Experimental Results

### Binary Evaluation

```bash
python Code/train_binary.py
```

Outputs:

```text
Results/Binary/
```

---

### Multiclass Evaluation

```bash
python Code/train_multiclass.py
```

Outputs:

```text
Results/Multiclass/
```

---

### MAIL Ablation

```bash
python Code/mail_ablation.py
```

Outputs:

```text
Results/MAIL_Ablation/
```

---

### Figure Generation

```bash
python Code/generate_figures.py
```

Outputs:

```text
Results/Figures/
```

---

# Saved Models

All trained models are stored in:

```text
Trained-models/
```

including:

* Binary Detection Models
* Multiclass Classification Models
* MAIL Ablation Models

---

# Citation

If you use CAVALRY-AI in your research, please cite:

```bibtex
@article{cavalryai2026,
  title={CAVALRY-AI: Cybersecurity Agentic Vision-Action Large Language Model with Risk-aware hYbrid Intelligence},
  author={Urooj, Misha},
  journal={Under Review},
  year={2026}
}
```

---

# License

Apache License 2.0

See LICENSE for details.

---

# Repository

GitHub:

https://github.com/mishaurooj/CAVALRY-AI
