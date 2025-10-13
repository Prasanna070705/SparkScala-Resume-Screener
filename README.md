# Spark-Powered Resume Screening Pipeline 📄✨

This project implements an end-to-end pipeline for screening and classifying resumes using Apache Spark. The system automates the process from raw resume images to a final classification, leveraging Optical Character Recognition (OCR), Natural Language Processing (NLP) for feature extraction, and a fine-tuned Transformer model for categorization.

The pipeline is designed to be scalable and robust, handling large volumes of resumes in a distributed environment.

---

## 🏛️ Project Architecture

The screening process is a multi-stage pipeline where the output of one stage serves as the input for the next.

1.  **🖼️ Stage 1: OCR Processing (`SparkResumeOCR.scala`)**
    * **Input**: A directory of resume images (JPG, PNG, etc.) stored in HDFS.
    * **Process**: A Spark job reads the image paths, distributes them across worker nodes, and applies Tesseract OCR to extract raw text from each image.
    * **Output**: A CSV file containing the filename and the extracted raw text.

2.  **🧹 Stage 2: NLP Data Cleaning & Feature Extraction (`SparkNERDataCleaning.scala`)**
    * **Input**: The CSV file with raw resume text from the OCR stage.
    * **Process**: This Spark job uses the **Spark NLP** library to perform Named Entity Recognition (NER) on the resume text. It identifies and extracts key sections like `Education`, `Experience`, `Certifications`, and `Skills`. It also enriches the skill set by matching extracted skills against a comprehensive ESCO (European Skills, Competences, Qualifications and Occupations) skills database.
    * **Output**: A cleaned and structured CSV file where each resume is broken down into its constituent sections (Education, Experience, etc.) and has a list of extracted skills.

3.  **🤖 Stage 3: Model Fine-Tuning & Classification**
    * **Input**: The structured CSV file from the NLP stage.
    * **Process**: The final stage involves fine-tuning a pre-trained RoBERTa model to classify resumes into different categories (e.g., "Data Scientist," "Java Developer"). The primary, fully implemented approach uses PySpark and the Hugging Face library. An alternative data preparation script is also provided in Scala using the Deep Java Library (DJL).
    * **Output**: A trained model capable of classifying new resumes, along with performance metrics.

---

## 💻 Technology Stack

* **Core Framework**: Apache Spark 3.5+
* **Languages**: Scala (for data processing) and Python (for model fine-tuning)
* **Data Processing**: Spark SQL, Spark Core
* **NLP/ML Libraries**:
    * **Spark NLP (John Snow Labs)**: For pre-trained NER models.
    * **Tess4J**: A Java wrapper for the Tesseract OCR engine.
    * **Hugging Face Transformers**: For loading and fine-tuning the RoBERTa model.
    * **PyTorch**: The backend for the Hugging Face model.
    * **Scikit-learn**: For label encoding and performance metrics.
* **Storage**: HDFS (Hadoop Distributed File System)

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

* `SparkResumeOCR.scala`: A Spark job that reads resume image files from HDFS, performs OCR using Tesseract, and saves the extracted text to a CSV.
* `SparkNERDataCleaning.scala`: Reads the raw text, applies a pre-trained NER model from Spark NLP to extract sections (Education, Experience, Skills), and saves the structured data.
* `pysparkrobertafinetune.py`: The primary script for Stage 3. It loads the structured data using PySpark, preprocesses it, and fine-tunes a `distilroberta-base` model using PyTorch and Hugging Face Transformers for resume classification.
* `RoBERTa_Final_Spark.scala`: An alternative Scala script showing the data preparation steps (text combination, label encoding, stratified splitting) for model training, with placeholder code for using the Deep Java Library (DJL).
