# HeronData: Enhanced File Classifier

## Overview

Efficient document processing is a cornerstone in industries like financial services, where accurate and scalable classification systems are essential. This project enhances the basic file classifier provided in the original repository, addressing key limitations such as:
1. Inability to handle poorly named files.
2. Difficulty in scaling to new industries with diverse document types.
3. Inefficiencies in processing large volumes of documents.

By leveraging advanced models like **BERT** and **DONUT**, this project implements a dual-classifier system:
- **Simple Classifier**: A semantic filename-based classifier, fine-tuned using BERT.
- **Advanced Classifier**: A content-based classifier leveraging the OCR-free DONUT model, built on the Swin Transformer architecture.

Both classifiers are deployed via a Flask-based frontend for testing. This README outlines the key features, installation steps, and future directions for this innovative classification system.

---

## Table of Contents

- [Features](#features)
- [Technical Background](#technical-background)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Development Guidelines](#development-guidelines)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## Features

### Simple Classifier
- **Semantic Filename Classification**: Utilizes a fine-tuned BERT model for filename-based classification.
- **Robust Handling**: Designed to manage ambiguities, typos, and unconventional filename formats.
- **Synthetic Filename Generator**: Generates diverse training data simulating real-world filename variations.

### Advanced Classifier
- **Content-Based Classification**: Employs the DONUT model to analyze document content directly.
- **OCR-Free Architecture**: Reduces dependency on OCR for extracting text from documents.
- **Augmented Dataset**: Trained on synthetic and real-world data enhanced with advanced augmentation techniques.

### Deployment and Frontend
- **AWS SageMaker Batch Transform**: Scalable deployment strategy optimized for processing large document batches.
- **Flask-Based Frontend**: Provides a user-friendly interface for uploading files and viewing classification results.

---

## Technical Background

### BERT
BERT (Bidirectional Encoder Representations from Transformers) is a natural language processing model that captures semantic nuances through bidirectional training, enabling the **Simple Classifier** to understand and categorize filenames effectively.

### DONUT Model
The **DONUT (Document Understanding Transformer)** model is a Swin Transformer-based architecture for document processing. Its OCR-free approach processes document images directly, making it ideal for the **Advanced Classifier**.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd HeronDataProject-dev
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix
   venv\Scripts\activate     # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Flask application to interact with the classifiers:
```bash
python -m src.app
```

---

## Configuration

Configuration files, including model paths and deployment settings, are located in the `src/config.py`. Environment variables can be set to customize the deployment.

---

## Testing

Testing is critical to validating the functionality and robustness of the classifiers. To run the test suite:
```bash
pytest testing/
```

Key test cases include:
- **Simple Classifier**: Validation of filename generation, classification accuracy, and performance metrics.
- **Advanced Classifier**: Evaluation of model loading, classification accuracy across diverse formats, and edge case handling.
- **Deployment Pipeline**: End-to-end tests of AWS SageMaker Batch Transform workflows.

---

## Deployment

The deployment strategy focuses on scalability and cost efficiency:
1. **AWS SageMaker Batch Transform**: Optimized for processing large-scale document classification tasks.
2. **Model Packaging**: The DONUT model is packaged as a `.tar.gz` archive for deployment on SageMaker.

Key AWS services used:
- **S3**: For storing input data and results.
- **Lambda**: For orchestrating workflows and monitoring S3 buckets.
- **API Gateway**: For exposing RESTful endpoints to interact with the classifiers.

---

## Development Guidelines

### Enhancements
- Expand the **Simple Classifier** dataset with additional filename variations.
- Train the **Advanced Classifier** on more diverse document layouts and languages.

### Refactoring
- Adopt modular design principles for better maintainability.
- Introduce abstraction layers for easier integration of additional models.

### CI/CD
- Implement automated testing and deployment pipelines using AWS CodePipeline and CodeBuild.

---

## Future Work

The project identifies several avenues for further enhancement:
1. **Feedback System**: Enable continuous learning by incorporating user feedback.
2. **Dataset Expansion**: Use real-world and synthetic data for improved generalization.
3. **Advanced Augmentation**: Leverage tools like Stable Diffusion for realistic document generation.
4. **Frontend Development**: Transition to a production-ready React Native application.
5. **Security Enhancements**: Incorporate AWS KMS for data encryption and refine IAM policies.
6. **Exploration of Alternative Models**: Evaluate ensemble approaches and other transformer architectures.

---

## Contributors

- **Shafir Rahman** - Development, Research, and Documentation.

Contributions are welcome! Submit a pull request or raise an issue for discussion.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## For details read the report!
