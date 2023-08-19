import pandas as pd
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE


############################ In this part we implement an algorithm to decrease the size of the class with more samples (Data Undersampling)####################


def sample_group(group, max_samples):
    if len(group) > max_samples:
        return group.sample(n=max_samples, random_state=123, axis=0)
    else:
        return group

def trim_dataframe(df, max_samples, min_samples, column):
    df = df.copy()
    groups = df.groupby(column)
    trimmed_df = pd.DataFrame(columns=df.columns)

    for label, group in groups:
        sampled_group = sample_group(group, max_samples)
        if len(sampled_group) >= min_samples:
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)

    return trimmed_df

def trim_dataset(df, max_samples, min_samples, column):
    print(f"Dataframe initially is of length {len(df)} with {df[column].nunique()} classes")
    
    trimmed_df = trim_dataframe(df, max_samples, min_samples, column)
    class_count = trimmed_df[column].nunique()

    print(f"After trimming, the maximum samples in any class is now {max_samples} and the minimum samples in any class is {min_samples}")
    print(f"The trimmed dataframe now is of length {len(trimmed_df)} with {class_count} classes")

    return trimmed_df, trimmed_df[column].unique(), class_count


###############################################################################################


#### Class Weighing is done already in pytorch loss function ####


######################################### In this part we implement oversampling SMOTHE method ########################################


def load_images_from_filepaths(filepaths):
    images = []
    for filepath in filepaths:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        images.append(img)
    return images

def perform_smote_oversampling(df, sampling_strategy='auto', random_state=42):
    """
    Perform SMOTE oversampling on the input DataFrame.

    Parameters:
        df (pandas DataFrame): DataFrame containing 'filepath' column for image file paths and 'label' column for labels.
        sampling_strategy (str, float or dict, optional): The ratio of the number of samples in the minority class 
                                                        over the number of samples in the majority class after resampling.
                                                        'auto' (default) uses a 1:1 ratio.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        augmented_df (pandas DataFrame): DataFrame containing augmented samples.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X = load_images_from_filepaths(df['filepath'].tolist())
    y = df['label'].tolist()

    X_resampled, y_resampled = smote.fit_resample(X, y)
    augmented_df = pd.DataFrame({'filepath': X_resampled, 'label': y_resampled})

    return augmented_df




# ############################################# Unsemble methods ####################################
# from sklearn.ensemble import VotingClassifier

# resnet_model = ...
# densenet_model = ...
# inception_model = ...
# X_train = ...
# y_train = ...
# X_test = ...
# y_test = ...

# # Assuming you have three models: resnet_model, densenet_model, and inception_model
# # Voting Ensemble:
# # In a Voting Ensemble, each model makes a prediction, and the final prediction is determined by a majority vote. The class with the most votes among the models is chosen as the final prediction.
# ensemble_model = VotingClassifier(estimators=[
#     ('resnet', resnet_model),
#     ('densenet', densenet_model),
#     ('inception', inception_model)
# ], voting='hard')

# # Train the ensemble model
# ensemble_model.fit(X_train, y_train)

# # Evaluate the ensemble model
# accuracy = ensemble_model.score(X_test, y_test)


# # Averaging Ensemble:
# # In an Averaging Ensemble, each model makes a prediction, and the final prediction is obtained by averaging the individual model predictions.
# # Assuming you have three models: resnet_model, densenet_model, and inception_model
# def ensemble_predict(models, X):
#     predictions = [model.predict(X) for model in models]
#     return np.mean(predictions, axis=0)

# models = [resnet_model, densenet_model, inception_model]

# # Make predictions using the ensemble model
# ensemble_predictions = ensemble_predict(models, X_test)

# # Convert the averaged predictions to class labels (if necessary)
# ensemble_class_labels = np.argmax(ensemble_predictions, axis=1)

# # Evaluate the ensemble model
# # accuracy = accuracy_score(y_test, ensemble_class_labels)
