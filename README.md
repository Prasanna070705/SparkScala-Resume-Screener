# Spark-Powered Resume Screening Pipeline 📄✨

This project implements an end-to-end pipeline for screening and classifying resumes using Apache Spark. The system automates the process from raw resume images to a final classification, leveraging Optical Character Recognition (OCR), Natural Language Processing (NLP) for feature extraction, and a fine-tuned Transformer model for categorization.

The pipeline is designed to be scalable and robust, handling large volumes of resumes in a distributed environment. Alongside the Spark jobs, the repository includes **Jupyter Notebooks** for each stage, which are useful for exploring the data, experimenting, and running the workflow interactively without a cluster.

---

## 🏛️ Project Architecture

The screening process is a multi-stage pipeline where the output of one stage serves as the input for the next.

1.  **🖼️ Stage 1: OCR Processing (`SparkResumeOCR.scala`)**
    * **Input**: A directory of resume images (JPG, PNG, etc.) stored in HDFS.
    * **Process**: A Spark job reads the image paths, distributes them across worker nodes, and applies Tesseract OCR to extract raw text from each image.
    * **Output**: A CSV file containing the filename and the extracted raw text.
    * **Notebook equivalent**: `datascraperimage2csv.ipynb`

2.  **🧹 Stage 2: NLP Data Cleaning & Feature Extraction (`SparkNERDataCleaning.scala`)**
    * **Input**: The CSV file with raw resume text from the OCR stage.
    * **Process**: This Spark job uses the **Spark NLP** library to perform Named Entity Recognition (NER) on the resume text. It identifies and extracts key sections like `Education`, `Experience`, `Certifications`, and `Skills`. It also enriches the skill set by matching extracted skills against a comprehensive ESCO (European Skills, Competences, Qualifications and Occupations) skills database.
    * **Output**: A cleaned and structured CSV file where each resume is broken down into its constituent sections (Education, Experience, etc.) and has a list of extracted skills.
    * **Notebook equivalent**: `ner-with-esco-dataset.ipynb`

3.  **🤖 Stage 3: Model Fine-Tuning & Classification**
    * **Input**: The structured CSV file from the NLP stage.
    * **Process**: The final stage involves fine-tuning a pre-trained RoBERTa model to classify resumes into different categories (e.g., "Data Scientist," "Java Developer"). The primary, fully implemented approach uses PySpark and the Hugging Face library. An alternative data preparation script is also provided in Scala using the Deep Java Library (DJL).
    * **Output**: A trained model capable of classifying new resumes, along with performance metrics.
    * **Notebook equivalent**: `complete-fine-tuning-of-roberta-with-completeness-2.ipynb`

---

## 💻 Technology Stack

* **Core Framework**: Apache Spark 3.5+
* **Languages**: Scala (for data processing) and Python (for model fine-tuning and notebooks)
* **Data Processing**: Spark SQL, Spark Core
* **NLP/ML Libraries**:
    * **Spark NLP (John Snow Labs)**: For pre-trained NER models.
    * **Tess4J**: A Java wrapper for the Tesseract OCR engine.
    * **Hugging Face Transformers**: For loading and fine-tuning the RoBERTa model.
    * **PyTorch**: The backend for the Hugging Face model.
    * **Scikit-learn**: For label encoding and performance metrics.
* **Notebooks**: Jupyter Notebook / JupyterLab
* **Storage**: HDFS (Hadoop Distributed File System)

---

## 📓 Jupyter Notebooks

The `src/Jupyter Notebooks/` folder contains three notebooks, one for each stage of the pipeline. They are a convenient way to inspect intermediate results, tweak parameters, and reproduce each stage on a single machine.

| Notebook | Pipeline Stage | What it does |
|---|---|---|
| `datascraperimage2csv.ipynb` | Stage 1: Data extraction | Extracts text from resume images and builds a consolidated resume dataset. Each resume becomes one row, with additional columns holding the information pulled from it. The resulting CSV feeds the NER stage. |
| `ner-with-esco-dataset.ipynb` | Stage 2: NER & skill enrichment | Performs Named Entity Recognition on the resume text using the **ESCO (European Skills, Competences, Qualifications and Occupations)** skills dataset, identifying and matching skills and structuring the resume into its sections. |
| `complete-fine-tuning-of-roberta-with-completeness-2.ipynb` | Stage 3: Model fine-tuning | A complete fine-tuning script for the RoBERTa model, which takes the **completeness** of each resume into account, so that performance can be analyzed by how complete a resume is (Low, Medium, High). |

### Running the notebooks

1.  Install Jupyter and the Python dependencies:

    ```bash
    pip install notebook
    pip install -r requirements.txt
    ```

2.  Launch Jupyter from the repository root and open the notebook you need from `src/Jupyter Notebooks/`:

    ```bash
    jupyter notebook
    ```

3.  Run the notebooks in order (`datascraperimage2csv` → `ner-with-esco-dataset` → `complete-fine-tuning-of-roberta-with-completeness-2`), since the output CSV of each one is the input of the next. Update any input and output file paths in the first cells to match your environment.

> **Note:** Notebooks that perform OCR also require the Tesseract OCR engine to be installed on your machine, and the fine-tuning notebook runs much faster on a GPU.

---

## 🚀 How to Run the Pipeline

### Prerequisites

* A working Apache Spark and Hadoop (HDFS) environment.
* Java and Scala installed.
* Python with PyTorch, Transformers, and other required libraries installed (`pip install -r requirements.txt`).
* Tesseract OCR engine installed on all Spark cluster nodes, along with the English `tessdata` pack.

### Execution Steps

1.  **Run OCR Processing**
    Place your resume images in the designated HDFS input directory. Then, submit the Scala Spark job.

    ```bash
    spark-submit --class SparkResumeOCR_HDFS \
                 --master <your_spark_master> \
                 SparkResumeOCR.jar
    ```

2.  **Run NER Data Cleaning**
    Once the OCR stage is complete, run the data cleaning and feature extraction job.

    ```bash
    spark-submit --class SparkNERDataCleaning \
                 --master <your_spark_master> \
                 --packages "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.0" \
                 SparkNERDataCleaning.jar
    ```

3.  **Run Model Fine-Tuning (PySpark)**
    Finally, run the Python script to fine-tune the RoBERTa model. The script reads the structured data and outputs a trained model.

    ```bash
    spark-submit pysparkrobertafinetune.py
    ```

> Prefer to work without a cluster? See the [Jupyter Notebooks](#-jupyter-notebooks) section above to run each stage interactively.

---

## Alternative Approach: Fine-Tuning with PySpark & Hugging Face

While the initial data processing stages are implemented in Scala for performance on the JVM, the most effective and flexible approach for fine-tuning modern Transformer models is within the Python ecosystem. The `pysparkrobertafinetune.py` script provides a complete and optimized solution for this final stage.

### Why this approach is powerful:

* **Leverages the Best of Both Worlds**: It uses Spark's powerful distributed data processing for loading and initial transformations and then seamlessly transitions to industry-standard Python libraries (Pandas, PyTorch, Hugging Face) for the modeling phase.
* **State-of-the-Art Models**: Gains direct access to thousands of pre-trained models from the Hugging Face Hub, like `distilroberta-base`.
* **Advanced Transfer Learning**: Implements sophisticated techniques like **layer freezing**, where only the top layers of the model are trained. This significantly speeds up training time and reduces computational cost while retaining high accuracy.
* **Configurable Speed vs. Accuracy**: The script includes multiple `OPTIMIZATION_LEVEL` presets ("FAST", "ULTRA_FAST", "BALANCED") that automatically adjust hyperparameters like batch size, model layers to freeze, and data filtering to meet different performance needs.
* **Robust Data Splitting**: It performs a **stratified split** based on both the resume `Category` and a calculated `completeness_bin`. This ensures that the training, validation, and test sets have a balanced representation of all data slices, preventing model bias.
* **Detailed Performance Analysis**: After training, the script evaluates the model's accuracy not just overall but also broken down by the completeness level of the resumes (Low, Medium, High). This provides crucial insights into how data quality affects prediction accuracy.

The Scala script `RoBERTa_Final_Spark.scala` serves as a proof-of-concept for how the data preparation could be done entirely within a Scala/DJL stack, but the **PySpark/Hugging Face script is the project's primary, fully functional, and recommended implementation for the model training stage**.

---

## 📂 File Descriptions

### Spark jobs and scripts

* `SparkResumeOCR.scala`: A Spark job that reads resume image files from HDFS, performs OCR using Tesseract, and saves the extracted text to a CSV.
* `SparkNERDataCleaning.scala`: Reads the raw text, applies a pre-trained NER model from Spark NLP to extract sections (Education, Experience, Skills), and saves the structured data.
* `pysparkrobertafinetune.py`: The primary script for Stage 3. It loads the structured data using PySpark, preprocesses it, and fine-tunes a `distilroberta-base` model using PyTorch and Hugging Face Transformers for resume classification.
* `RoBERTa_Final_Spark.scala`: An alternative Scala script showing the data preparation steps (text combination, label encoding, stratified splitting) for model training, with placeholder code for using the Deep Java Library (DJL).

### Jupyter Notebooks (`src/Jupyter Notebooks/`)

* `datascraperimage2csv.ipynb`: Extracts data from resume images and creates a consolidated resume dataset, with one row per resume and columns for the information extracted from it.
* `ner-with-esco-dataset.ipynb`: Performs Named Entity Recognition on resume text using the European ESCO skills dataset to identify and enrich skills.
* `complete-fine-tuning-of-roberta-with-completeness-2.ipynb`: The complete RoBERTa fine-tuning script, including resume completeness handling for classification.
