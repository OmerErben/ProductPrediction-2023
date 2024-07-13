# ProductPrediction

ProductPrediction is a data analysis and machine learning application designed to process and predict product categories using both tabular data and image data. 

The project includes models such as CNN (Convolutional Neural Network), XGBoost, Random Forest, and DNN (Deep Neural Network) to analyze and classify the data.

## Installation and Setup

### Install Git LFS

**On Windows:**
- Download and run the installer from the [Git LFS website](https://git-lfs.github.com/).

**On macOS:**
```bash
brew install git-lfs
```

**On Linux:**
```bash
sudo apt-get install git-lfs
```

### Initialize Git LFS
```bash
git lfs install
```

Clone the Repository
```bash
git clone https://github.com/MIKIHERSHCOVITZ/ProductPrediction.git
cd ProductPrediction
```

Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**Ensure Data Directory Structure:**

Make sure the data and images directories are structured as shown in the project structure section below.

**Running the Application**

```bash
python main.py
```



## Project Structure
```
ProductPrediction/
│
├── data/
│   ├── food_nutrients.csv
│   ├── food_test.csv
│   ├── food_train.csv
│   └── nutrients.csv
│
├── images/
│   └── train/
│       ├── class_1/
│       │   ├── image1.jpg
│       │   └── ...
│       ├── class_2/
│       │   ├── image1.jpg
│       │   └── ...
│       └── ...
│
├── models/
│   ├── cnn_model.py
│   ├── dnn_model.py
│   ├── rf_model.py
│   └── xgb_model.py
│
├── scripts/
│   ├── data_preprocessing.py
│   └── image_preprocessing.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Files and Directories

1. data/: Contains CSV files with tabular data for training and testing. 
2. images/: Contains directories of images for training, organized by class. 
3. models/: Contains Python scripts defining the machine learning models. 
4. scripts/: Contains Python scripts for data preprocessing and image preprocessing. 
5. main.py: Main script to run the project. 
6. requirements.txt: Lists the Python dependencies required for the project. 
7. .gitignore: Specifies files and directories to be ignored by Git. 
8. README.md: Documentation for the project.

## Functionality

1. Data Preprocessing: Loads and preprocesses tabular data from CSV files. 
2. Image Preprocessing: Loads and preprocesses images from directories. 
3. Model Training: Trains various machine learning models including CNN, XGBoost, Random Forest, and DNN. 
4. Evaluation: Evaluates the trained models on test data and prints classification reports.

## Prerequisites

Python 3.8 or higher

Virtualenv

TensorFlow 2.x


## Usage

Data Preprocessing: The data_preprocessing.py script loads and preprocesses the tabular data.

Image Preprocessing: The image_preprocessing.py script loads and preprocesses the image data.

Model Training and Evaluation: The main.py script trains and evaluates the models.