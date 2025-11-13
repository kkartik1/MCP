import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv('sample.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df['Label'].value_counts())
print("\nFirst few rows:")
print(df.head())

# Split the data into train (80%) and test (20%) sets
# stratify ensures the same proportion of labels in both sets
# random_state ensures reproducibility
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['Label']
)

# Save to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Display split information
print("\n" + "="*50)
print("Split completed successfully!")
print("="*50)
print(f"\nTraining set size: {len(train_df)} samples")
print(f"Test set size: {len(test_df)} samples")

print("\nTraining set label distribution:")
print(train_df['Label'].value_counts())

print("\nTest set label distribution:")
print(test_df['Label'].value_counts())

print("\nFiles saved:")
print("- train.csv")
print("- test.csv")