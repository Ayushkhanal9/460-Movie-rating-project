import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import scipy.sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Load the data
data = pd.read_csv("dataprep/filtered_reviews.csv")

# Convert 'review_score' column to numeric type
def convert_fraction_to_float(fraction_str):
    try:
        numerator, denominator = map(int, fraction_str.split('/'))
        return numerator / denominator
    except ValueError:
        return None

data['review_score'] = data['review_score'].apply(convert_fraction_to_float)

# Drop rows with NaN values in 'review_score' column
data = data.dropna(subset=['review_score'])

# Map the 'review_score' to the nearest 0.2
data['review_score'] = data['review_score'].map(lambda x: round(x * 5) / 5)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.05, random_state=40)

# Text cleaning: Remove unnecessary words (stopwords)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

train_data['review_content'] = train_data['review_content'].apply(clean_text)
test_data['review_content'] = test_data['review_content'].apply(clean_text)

# Tokenize the review content using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['review_content']).toarray()
X_test = vectorizer.transform(test_data['review_content']).toarray()

# Convert the review scores to PyTorch tensors
y_train = torch.tensor(train_data['review_score'].values, dtype=torch.float32)
y_test = torch.tensor(test_data['review_score'].values, dtype=torch.float32)

# Define the Feedforward Neural Network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Create a custom dataset and DataLoader
class ReviewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

train_dataset = ReviewsDataset(X_train, y_train)
test_dataset = ReviewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 128
output_size = 1
model = FFNN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# Test the model and calculate accuracy for each label
model.eval()
with torch.no_grad():
    correct_per_label = {0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1.0: 0}
    total_per_label = {0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0, 1.0: 0}
    total_correct = 0
    total_samples = 0
    test_loss = 0.0

    for inputs, labels in tqdm(test_loader, desc='Testing'):
        outputs = model(inputs)
        test_loss += criterion(outputs, labels.view(-1, 1))

        # Convert predictions to the nearest 0.2
        predicted_labels = torch.round(outputs * 5) / 5

        # Calculate accuracy for each label
        for label, predicted_label in zip(labels.view(-1), predicted_labels.view(-1)):
            rounded_label = round(label.item(), 1)
            total_per_label[rounded_label] += 1

            rounded_predicted_label = round(predicted_label.item(), 1)
            if rounded_predicted_label == rounded_label:
                correct_per_label[rounded_label] += 1
                total_correct += 1

        total_samples += labels.size(0)

    test_loss /= len(test_loader)

    # Print overall test loss
    print(f'Test Loss: {test_loss.item()}')

    # Print accuracy for each label
    for label in sorted(correct_per_label.keys()):
        accuracy = correct_per_label[label] / total_per_label[label] if total_per_label[label] > 0 else 0.0
        print(f'Accuracy for label {label:.1f}: {accuracy * 100:.2f}%')

    # Print overall accuracy
    overall_accuracy = total_correct / total_samples
    print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')

