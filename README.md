###INSTALLING DEPENDENCIES AND LIBRARIES###

pip install streamlit numpy pillow scikit-image tensorflow requests

###VERSION REQUIREMENTS###

Python 3.7 or higher
streamlit >= 1.0.0
numpy >= 1.19.0
scikit-image >= 0.18.0
tensorflow >= 2.0.0
pillow >= 8.0.0
requests >= 2.25.0

###HOW TO RUN###

*to be run in terminal(ensure the terminal is running in the same file path)
streamlit run HACKATHON-streamlit.py

###Project Overview###
        The AI Image Classification Hackathon 2025 is a challenge focused on building an image classification model to identify and categorize waste. Participants will go through the entire process of developing and deploying a machine learning solution, including data loading and preprocessing, model training and evaluation, deployment on cloud platforms, API integration, and UI integration.   
The project aims to develop a solution for smart waste management using AI.
     Dataset used:
     Dataset provided was downloaded from the link provided on Waste Image Classification Dataset from https://prod-dcd datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n3gtgm9jxj-2.zip

###devlopment process###
Developed by #Pempho katsala

##importing libraries

<img width="353" alt="importing libraries" src="https://github.com/user-attachments/assets/0cfaf2fa-6acf-428a-a946-085320df2276" />

##importing the data set

<img width="500" alt="importing ZIP" src="https://github.com/user-attachments/assets/282be83c-8ec9-42d9-96f0-2447a0411175" />

##extracting the data set

<img width="442" alt="Extracting_the_dataset" src="https://github.com/user-attachments/assets/1eda5484-8eb9-4449-a162-a8f3505767b9" />

##script used for data cleaning

```python
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.impute import SimpleImputer

def clean_data(dataset_paths):
    """Handles duplicates, missing data, outliers, and inconsistent data in image datasets."""

    all_dfs = []  # List to store DataFrames from each path

    for dataset_path in dataset_paths:
        # Check if path exists
        if not os.path.exists(dataset_path):
            print(f"Warning: Path {dataset_path} does not exist. Skipping...")
            continue

        # Get all image files
        try:
            image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        except Exception as e:
            print(f"Error accessing directory {dataset_path}: {e}")
            continue

        # Create dataframe
        data = {'filename': image_files, 'filepath': [os.path.join(dataset_path, f) for f in image_files]}
        df = pd.DataFrame(data)

        if df.empty:
            print(f"No images found in {dataset_path}. Skipping...")
            continue

        print(f"Processing {len(df)} images from {dataset_path}")

        # 1. Handle Duplicates (based on filename)
        before_drop = len(df)
        df.drop_duplicates(subset='filename', keep='first', inplace=True)
        print(f"  - Removed {before_drop - len(df)} duplicate files")

        # 2. Handle Corrupted Images
        df['corrupted'] = False  # Initialize column properly
        corrupted_count = 0

        for index, row in df.iterrows():
            try:
                img = Image.open(row['filepath'])
                img.verify()  # Verify image
                # Also try to load it to catch other potential issues
                img = Image.open(row['filepath'])
                img.load()
            except Exception as e:
                df.at[index, 'corrupted'] = True
                corrupted_count += 1

        print(f"  - Identified {corrupted_count} corrupted images")

        # 3. Handle Missing Data (image dimensions)
        dimensions = []
        for filepath in df['filepath']:
            try:
                if os.path.exists(filepath):
                    img = Image.open(filepath)
                    width, height = img.size
                    dimensions.append((width, height))
                else:
                    dimensions.append((None, None))
            except Exception:
                dimensions.append((None, None))

        df[['width', 'height']] = pd.DataFrame(dimensions, index=df.index)

        # Count missing values before imputation
        missing_width = df['width'].isna().sum()
        missing_height = df['height'].isna().sum()
        print(f"  - Missing dimensions: {missing_width} width, {missing_height} height")

        # Impute missing dimensions if there are any non-missing values
        if not df[['width', 'height']].isna().all().all():
            imputer = SimpleImputer(strategy='median')
            df[['width', 'height']] = imputer.fit_transform(df[['width', 'height']])
            print(f"  - Imputed missing dimensions with median values")

        # 4. Handle Outliers (image dimensions)
        before_outlier = len(df)

        # Only process if we have enough data for meaningful quartiles
        if len(df) > 10:
            # Function to detect and mark outliers
            def mark_outliers(df, column):
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                return ~((df[column] >= lower_bound) & (df[column] <= upper_bound))

            # Mark outliers for both dimensions
            df['width_outlier'] = mark_outliers(df, 'width')
            df['height_outlier'] = mark_outliers(df, 'height')

            # Remove rows where both width and height are outliers
            outliers = df['width_outlier'] & df['height_outlier']
            df = df[~outliers]

            # Clean up the temporary columns
            df = df.drop(['width_outlier', 'height_outlier'], axis=1)

            print(f"  - Removed {before_outlier - len(df)} outlier images")

        # 5. Handle Inconsistent Data (filename case)
        df['filename'] = df['filename'].str.lower()

        # Remove corrupted images from final dataset
        before_corrupt_removal = len(df)
        df = df[df['corrupted'] == False]
        print(f"  - Removed {before_corrupt_removal - len(df)} corrupted images")

        # Drop the corrupted column
        df = df.drop('corrupted', axis=1)

        print(f"  - Final count: {len(df)} clean images\n")
        all_dfs.append(df)  # Append the cleaned DataFrame to the list

    # Concatenate all DataFrames if we have any
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total clean images across all directories: {len(combined_df)}")
        return combined_df
    else:
        print("No valid images found in any of the provided paths.")
        return pd.DataFrame()  # Return empty DataFrame if no valid data

# Add a category label based on the directory
def add_category_labels(df):
    """Add category labels based on the filepath"""
    df['category'] = df['filepath'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    return df

def save_cleaned_data(cleaned_df, output_base_dir, target_size=(256, 256)):
    """
    Save cleaned data to a new directory structure, preserving categories and standardizing images.

    Args:
        cleaned_df: DataFrame with 'filepath' and 'filename' columns
        output_base_dir: Base directory to save cleaned data
        target_size: Tuple of (width, height) to resize images to

    Returns:
        Dictionary mapping original categories to new save paths
    """
    # Create the base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Created output directory: {output_base_dir}")

    # Create a dictionary to track categories and their save paths
    category_paths = {}
    saved_count = 0
    error_count = 0

    # For each file in the cleaned DataFrame
    for idx, row in cleaned_df.iterrows():
        try:
            # Extract category from original filepath
            category = os.path.basename(os.path.dirname(row['filepath']))

            # Create category directory if it doesn't exist in our tracking dict
            if category not in category_paths:
                category_dir = os.path.join(output_base_dir, category)
                os.makedirs(category_dir, exist_ok=True)
                category_paths[category] = category_dir
                print(f"Created category directory: {category_dir}")

            # Load the original image
            img = Image.open(row['filepath'])

            # Standardize to desired format (RGB, specific size)
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Define save path
            save_path = os.path.join(category_paths[category], row['filename'])

            # Save the image
            img.save(save_path)
            saved_count += 1

            # Print progress update for every 100 images
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} images...")

        except Exception as e:
            print(f"Error saving {row['filepath']}: {e}")
            error_count += 1

    print(f"Successfully saved {saved_count} cleaned images")
    if error_count > 0:
        print(f"Encountered errors with {error_count} images")

    # Report counts per category
    for category, path in category_paths.items():
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
        print(f"Category '{category}': {count} images")

    return category_paths

# Main function to execute the entire workflow
def main():
    # Define the paths to your dataset directories
    filepath1 = r'/content/dataset/extracted/Waste Classification Dataset/waste_dataset/organic'
    filepath2 = r'/content/dataset/extracted/Waste Classification Dataset/waste_dataset/recyclable'

    # Define output directory for cleaned data
    cleaned_output_dir = "/content/cleaned_waste_dataset"

    # Check if directories exist
    print("Checking dataset paths...")
    for path in [filepath1, filepath2]:
        if os.path.exists(path):
            print(f"Path exists: {path}")
            print(f"Contains {len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])} images")
        else:
            print(f"Path does not exist: {path}")

    # Only proceed with paths that exist
    valid_paths = [path for path in [filepath1, filepath2] if os.path.exists(path)]

    if valid_paths:
        print("\n=== STEP 1: Cleaning the dataset ===")
        cleaned_df = clean_data(valid_paths)

        if not cleaned_df.empty:
            print("\n=== STEP 2: Adding category labels ===")
            labeled_df = add_category_labels(cleaned_df)
            print("Category distribution:")
            print(labeled_df['category'].value_counts())

            print("\n=== STEP 3: Saving standardized images ===")
            # Save cleaned and standardized images
            category_paths = save_cleaned_data(labeled_df, cleaned_output_dir, target_size=(256, 256))

            print("\n=== Process Complete ===")
            print(f"Cleaned data saved to: {cleaned_output_dir}")
            print("The cleaned dataset is now ready for model training")
        else:
            print("No valid images found after cleaning. Please check your dataset.")
    else:
        print("No valid paths to process. Please check your directory structure.")

# Execute the main function
if __name__ == "__main__":
    main()
```
##data cleaning

the data was cleaned using the following techniques:

 *df.drop_duplicates(subset='filename', keep='first', inplace=True) to remove duplicates based on the filename. 

 *Added a dummy_metadata column to the dataframe, and populated it with some NaN values.#

 *SimpleImputer from scikit-learn to handle missing values in the dummy_metadata column.

*Added a dummy_feature column that Uses a simple 3-standard-deviation rule to remove outliers.
*To Combine the processed DataFrames from all paths into a single DataFrame pd.concat() is used.

##script used for data preprocessing

``` python

import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.color import rgb2gray

def extract_features(image_path):
    """Extract HOG features from 256x256 normalized images"""
    try:
        # Load and standardize image size
        img = load_img(image_path, target_size=(256, 256))

        # Normalize pixel values to [0,1]
        img_array = img_to_array(img) / 255.0

        # Convert to grayscale for HOG
        img_gray = rgb2gray(img_array)

        # Extract HOG features
        return hog(img_gray,
                 orientations=8,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2),
                 transform_sqrt=True)
    except Exception as e:
        raise RuntimeError(f"Error processing {image_path}: {str(e)}")

def build_dataset(base_dir):
    """Create structured dataset from directory"""
    image_data = []
    class_counts = {'organic': 0, 'recyclable': 0}

    for class_name in ['organic', 'recyclable']:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Missing directory: {class_dir}")

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_data.append({
                    'path': os.path.join(class_dir, fname),
                    'label': class_name
                })
                class_counts[class_name] += 1

    print("Dataset composition:")
    print(f"  Organic: {class_counts['organic']} images")
    print(f"  Recyclable: {class_counts['recyclable']} images")
    return pd.DataFrame(image_data)

# Main processing workflow
base_dir = "/content/cleaned_waste_dataset"
df = build_dataset(base_dir)

# Feature extraction
features = []
labels = []
valid_paths = []

print("\nFeature extraction progress:")
for idx, row in df.iterrows():
    try:
        features.append(extract_features(row['path']))
        labels.append(row['label'])
        valid_paths.append(row['path'])
    except Exception as e:
        print(f"Skipped {row['path']}: {str(e)}")

# Convert to numpy arrays
features_array = np.array(features)
labels_array = LabelEncoder().fit_transform(labels)

# Save processed data
output_dir = "/content/hackathon_features"
os.makedirs(output_dir, exist_ok=True)

np.savez(os.path.join(output_dir, 'processed_data.npz'),
         features=features_array,
         labels=labels_array)

pd.DataFrame({'path': valid_paths, 'label': labels})\
  .to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

print("\nProcessing completed!")
print(f"Final dataset size: {features_array.shape[0]} samples")
print(f"Feature vector length: {features_array.shape[1]}")
print(f"Saved to: {output_dir}")

```

###data preprocessing

the techniques used in the preprocessing stage were:

*load_img(image_path, target_size=(256, 256)): Images are loaded and resized to a consistent 256x256 pixel dimension. This standardizes the input size for feature extraction.

*img_to_array(img) / 255.0: Pixel values are normalized to the range [0, 1] by dividing by 255.0.

*rgb2gray(img_array): The color image is converted to grayscale. This is necessary because the HOG (Histogram of Oriented Gradients) feature extractor works on grayscale images.

*hog(...): HOG features are extracted from the grayscale images. HOG is a feature descriptor wich captures the distribution of gradient orientations in localized portions of an image.

*Parameters used inside the hog function:

~orientations=: Number of orientation bins.

~pixels_per_cell=(8, 8): Size of each cell.

~cells_per_block=(2, 2): Number of cells in each block.

~transform_sqrt=True: Power law compression.

*LabelEncoder().fit_transform(labels): Categorical labels ('organic', 'recyclable') are converted into numerical representations (0, 1). 

###Training the model###

##loading the X and Y variables

<img width="307" alt="Screenshot 2025-04-02 141903" src="https://github.com/user-attachments/assets/b0e5fcac-48dc-4c89-ac5a-5d463bd33fc1" />

the code loads x and y from the pre-processed dataframe with x being features and y being labels

##trainig the logistic regression model

<img width="379" alt="LR training" src="https://github.com/user-attachments/assets/367c6678-4a27-4bfc-a452-83db8c0df207" />

#imported libraries

*sklearn.model_selection.train_test_split: This function is used to split the dataset into training and testing subsets, which is crucial for evaluating the model's performance.

*sklearn.linear_model.LogisticRegression: This is the class for the Logistic Regression model, a linear model used for binary classification.

*joblib: This library is used for efficient saving (serialization) and loading (deserialization) of Python objects, including trained machine learning models.

#Model Initialization:

*LogisticRegression(...): Creates an instance of the Logistic Regression model.

*max_iter=200: Sets the maximum number of iterations for the solver to converge.(default iter is 100)

*solver='lbfgs': Specifies the optimization algorithm used to find the model's coefficients.#Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm is great for limited resources

*train_test_split(X, y, train_size=0.8): Splits the dataset into training and testing sets. train_size=0.8: Specifies that 80% of the data should be used for training, and the remaining 20% for testing.

*model.fit(X_train, y_train): Trains the Logistic Regression model using the training data (X_train, y_train). 

*model.predict(X_test): Uses the trained model to make predictions on the test data (X_test). 

*joblib.dump(model, 'LR_model.pkl'): Saves the trained model to a file named 'LR_model.pkl'.

 ##testing the model's accuracy

<img width="388" alt="accuracy" src="https://github.com/user-attachments/assets/7a1fc050-eb93-4ec4-a84a-a3899b560a28" />

#our model managed to reach an accuracy of 94.33%

 ###Deployment
 
 ##the following files must be prepared in advace
 
*score.py
*config.json


#script for score.py

 ``` python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('LR_model')  
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data).reshape(1, -1)

        # Perform prediction using the loaded scikit-learn model
        result = model.predict(data)

        # You can return the result as a dictionary or in any desired format
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})

 ```
#the score.py file defines how the model will recieve inputs
score.py does:

 *Loads a trained Logistic Regression model from Azure ML's model registry upon service initialization.
 
 *Defines a run() function that receives JSON data, parses it, reshapes it for prediction, and returns the prediction result as a JSON response.

##script for config.json

``` 
{
    "subscription_id": "####-#####-####",
    "resource_group": "####",
    "workspace_name": "hackathon",
    "region": "eastus"
}

```
#the config.json file consits of the information used by azure to load the model

#script for deploying on microsoft azure

``` python

!pip install azureml-sdk

import os
import json
import requests

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

# laoding the configuration file - standard way - use .env file and load_dotenv from python-dotenv module
config_file_path = "/home/azureuser/cloudfiles/code/Users/pemphokatsala/hackathon ai/config.json"

# Read JSON data into a dictionary
with open(config_file_path, 'r') as file:
    data = json.load(file)

subscription_id = data["subscription_id"]
resource_group = data["resource_group"]  
workspace_name = data["workspace_name"]
region = data["region"]

ws = Workspace.create(name=workspace_name,
                      subscription_id=subscription_id,
                      resource_group=resource_group,
                      location=region)

print(f'Workspace {workspace_name} created')

# Specify the path to your  model file
model_path = '/home/azureuser/cloudfiles/code/LR_model.pkl'
model_name='LR_model'

# Register the model in Azure Machine Learning
registered_model = Model.register(model_path=model_path, model_name=model_name, workspace=ws)

# Create a Conda environment for your scikit-learn model
conda_env = Environment('my-conda-env')
conda_env.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])

# Create an InferenceConfig
inference_config = InferenceConfig(entry_script='/home/azureuser/cloudfiles/code/Users/pemphokatsala/hackathon ai/score.py', environment=conda_env)

# Specify deployment configuration for ACI
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name='lr-prediction',
                       models=[registered_model],
                       inference_config=inference_config,
                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)

scoring_uri = service.scoring_uri
scoring_uri

```
Setup and Configuration:

Installs the Azure ML SDK.
Imports necessary Azure ML libraries.
Reads Azure workspace configuration details (subscription ID, resource group, workspace name, region) from a JSON file (config.json).
Creates or retrieves an Azure Machine Learning workspace based on the configuration.
Model Registration:

Specifies the path to the pre-trained Logistic Regression model file (LR_model.pkl).
Registers the model in the Azure ML workspace, making it available for deployment.
Environment Setup:

Creates a Conda environment specifically for the model's dependencies.
Adds scikit-learn as a required package to the Conda environment, ensuring the model can run correctly in the deployed service.
Inference Configuration:

Creates an InferenceConfig object, which defines how the model will be used for inference.
Specifies the entry script (score.py), which contains the code for loading the model and making predictions.
Associates the previously created Conda environment with the inference configuration.
Deployment Configuration:

Creates an AciWebservice.deploy_configuration object to specify the deployment settings for Azure Container Instances.
Sets the CPU cores and memory allocation for the ACI container.
Model Deployment:

Deploys the registered model as a web service using Model.deploy().
Specifies the workspace, service name (lr-prediction), model to deploy, inference configuration, and deployment configuration.
Waits for the deployment to complete and displays the deployment output.
Retrieving Scoring URI:

Retrieves the scoring URI of the deployed web service. This URI is the endpoint that can be used to send prediction requests to the model.

###User interface

#script for creating user intrerface

 ``` python

import streamlit as st
import os
import requests
import logging
import numpy as np
from PIL import Image
import json
from skimage.feature import hog
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
AZURE_ENDPOINT = "http://6e0448db-282e-4f7c-b8b6-2b4f8e147e6b.eastus.azurecontainer.io/score"

def extract_features(image_path):
    """Extract HOG features to match training preprocessing"""
    try:
        # Load and standardize image size (matching training)
        img = load_img(image_path, target_size=(256, 256))
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Convert to grayscale for HOG
        img_gray = rgb2gray(img_array)
        
        # Extract HOG features (matching training parameters)
        features = hog(img_gray,
                      orientations=8,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      transform_sqrt=True)
        
        return features
        
    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}")
        raise

def validate_endpoint():
    """Validate if the Azure endpoint is accessible."""
    try:
        headers = {"Content-Type": "application/json"}
        # Create sample data matching the expected feature size
        sample_features = np.zeros(30752)  # HOG feature size from training
        test_data = json.dumps({"data": [sample_features.tolist()]})
        response = requests.post(AZURE_ENDPOINT, headers=headers, data=test_data)
        
        logging.info(f"Endpoint validation status code: {response.status_code}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Endpoint validation failed with status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Azure endpoint is not accessible: {str(e)}")
        logging.error(f"Azure endpoint connection error: {str(e)}")
        return False

def send_image_to_azure(image_path):
    """Send processed image data to Azure endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        # Extract HOG features
        processed_data = extract_features(image_path)
        
        # Create payload with processed features
        payload = {
            "data": [processed_data.tolist()]
        }

        logging.info(f"Sending request to Azure...")
        logging.info(f"Feature vector length: {len(processed_data)}")

        # Make request
        response = requests.post(
            AZURE_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )

        logging.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logging.info(f"Prediction response: {result}")
            
            # Map numerical predictions to labels
            label_map = {0: "Organic", 1: "Recyclable"}
            prediction = result.get("result", [0])[0]
            return {"prediction": label_map.get(prediction, "Unknown")}
        else:
            logging.error(f"Request failed: {response.text}")
            raise requests.HTTPError(f"Request failed with status {response.status_code}")
            
    except Exception as e:
        logging.error(f"Request failed: {str(e)}", exc_info=True)
        return None

def main():
    st.title("Image Upload and Azure Processing")

    uploaded_file = st.file_uploader("Upload an image for processing", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            temp_image_path = os.path.join("temp_image", uploaded_file.name)
            os.makedirs("temp_image", exist_ok=True)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"Image saved to {temp_image_path}.")

            with st.spinner("Sending image to Azure..."):
                response = send_image_to_azure(temp_image_path)

                if response:
                    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
                    st.markdown("**Azure Response:**")
                    st.json(response)
                else:
                    st.error("Failed to process the image.")

            os.remove(temp_image_path)
            logging.info(f"Temporary image {temp_image_path} deleted.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()


 ```

1. Imports and Logging:

Imports necessary libraries like streamlit, requests, PIL, numpy, skimage, and tensorflow.

Configures logging to record information and errors.

Defines the Azure endpoint URL as a constant.

2. extract_features(image_path) Function:

Loads an image from the given path.

Resizes the image to 256x256 pixels.

Converts the image to a NumPy array and normalizes pixel values.

Converts the color image to grayscale.

Extracts HOG features from the grayscale image, using the same parameters as the model training.

Returns the extracted HOG features.


Includes error handling with logging.

3. validate_endpoint() Function:

Sends a test request to the Azure endpoint to ensure it's accessible.

Creates a sample feature vector filled with zeros.

Sends a POST request with the sample data to the Azure endpoint.

Checks the response status code.

Displays error messages or logs the endpoint validation status.

Returns True if the endpoint is valid, False otherwise.

4. send_image_to_azure(image_path) Function:

Extracts HOG features from the uploaded image.

Constructs a JSON payload containing the extracted features.

Sends a POST request to the Azure endpoint with the payload.

Parses the JSON response from the endpoint.

Maps the numerical prediction from the Azure endpoint to human readable labels.

Handles potential errors during the request and response processing.

Logs the request and response details.

5. main() Function:

Creates a Streamlit title and file uploader.


#If an image is uploaded:

Saves the uploaded image to a temporary file.

Displays a spinner while the image is being processed and sent to Azure.

Calls send_image_to_azure() to send the image data to the Azure endpoint.

If a valid response is received:

Displays the uploaded image.

Displays the prediction result from Azure in JSON format.

If no valid response is received, displays an error message.

Deletes the temporary image file.

Handles potential exceptions during the process.

If no image is uploaded, displays an info message.





 






