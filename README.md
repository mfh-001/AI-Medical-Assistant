# MediScan AI: Multimodal Medical Diagnostic Assistant
This repository features **MediScan AI**, an advanced medical assistant that combines Computer Vision (CV) for skin lesion segmentation and Large Language Models (LLMs) for conversational health diagnostics. The project integrates a U-Net architecture for medical imaging and a Knowledge Base-augmented Llama 3.2 model to provide patient-centric medical guidance.

## Overview
The deployment-ready system provides an end-to-end diagnostic pipeline through an interactive Streamlit interface:
- **Dermatological Analysis**: Utilizing a custom-trained **U-Net** model to segment skin lesions and calculate affected area percentages from user-uploaded images.
- **Conversational AI**: A RAG-inspired (Retrieval-Augmented Generation) medical chatbot powered by **Llama 3.2 3B Instruct** for patient education.
- **Robust UI/UX**: A real-time dashboard that handles noisy user input and provides formatted medical summaries.

## Engineering Logic

### 1. Image Segmentation (U-Net)
To isolate skin lesions, the system implements a U-Net architecture. It processes $256 \times 256$ RGB images, producing a binary mask where each pixel is classified as 'Lesion' or 'Background'. The lesion severity is quantified by:

$$\text{Area Percentage} = \left( \frac{\sum \text{Pixels}_{\text{mask}}}{\sum \text{Pixels}_{\text{total}}} \right) \times 100$$

The model was trained on the `HAM10000` dataset, optimizing for the Dice Coefficient to handle class imbalance between healthy skin and lesion pixels.


### 2. LLM Instruction Tuning & Inference
After experimenting with specialized medical models, the system utilizes **Llama 3.2 3B** via 4-bit quantization (BitsAndBytes). This allows for efficient local inference while maintaining high conversational quality.
- **Text Normalization**: Custom fuzzy matching with a **75% similarity threshold** to handle common medical misspellings (e.g., "diabetis" $\rightarrow$ "diabetes").
- **Knowledge Augmentation**: The model is prompted with a localized medical dictionary to ensure responses remain grounded in factual clinical definitions rather than hallucinations.

## Project Challenges & Evolution
The development of MediScan AI involved overcoming significant hurdles in model selection and data behavior:
* **The "Academic" Trap**: Initial attempts with **BioGPT** and **Meditron-7B** failed because they were trained on research abstracts. They generated clinical papers instead of talking to patients. Switching to **Llama 3.2 3B Instruct** provided the necessary conversational empathy.
* **Catastrophic Forgetting**: Fine-tuning BioGPT on PubMedQA led to the model memorizing training samples verbatim, losing its ability to generalize.
* **Segmentation Accuracy**: Initial U-Net training showed a loss plateau at **0.237**. This was mitigated by enhancing the preprocessing pipeline and normalizing image intensities across the `HAM10000` dataset.

## Tech Stack
- **Deep Learning:** PyTorch, Transformers, PEFT (LoRA)
- **Computer Vision:** OpenCV, PIL
- **Large Language Models:** Meta Llama 3.2 3B (via Hugging Face)
- **Deployment:** Streamlit, ngrok (for interactive tunneling)
- **Libraries:** `fuzzywuzzy` (text matching), `bitsandbytes` (quantization)

## 📂 Project Structure
- **MediScan AI (medical assistant).ipynb**: The core notebook containing model training, inference logic, and the Streamlit app code.
- **requirements.txt**: List of necessary dependencies (attached in repo).
- **MediScan AI - Demo Video.mov**: A walkthrough of the functional application (attached in repo).

## Visual Results
The final application features a clean, professional interface. It provides a visual comparison of skin lesions while separately sharing the text-based query response in plain English.

<img width="1109" height="584" alt="Screenshot 2026-01-11 at 8 14 01 PM" src="https://github.com/user-attachments/assets/2b41c53e-a90b-471c-9d9b-8c1cfe20cb5c" />

---

## ⚠️ Academic Integrity
This repository is intended solely as a piece to showcase my learning journey. 
If you are a student working on a similar assignment: **do not copy this code.** Plagiarism is a serious offense that can lead to expulsion. Use this only as a conceptual reference.
