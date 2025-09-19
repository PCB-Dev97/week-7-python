# week-7-python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("=" * 50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 50)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Explore structure
    print("\nDataset information:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Clean dataset (though Iris dataset typically has no missing values)
    if df.isnull().sum().sum() > 0:
        print("\nCleaning dataset...")
        # Fill numerical columns with mean, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("Missing values handled!")
    else:
        print("\nNo missing values found. Dataset is clean!")
        
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "=" * 50)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 50)

# Basic statistics
print("Basic statistics for numerical columns:")
print(df.describe())

# Statistics by species
print("\nStatistics grouped by species:")
species_stats = df.groupby('species').describe()
print(species_stats)

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis
print("\nAdditional analysis:")
print(f"Number of samples per species:")
print(df['species'].value_counts())

# Interesting findings
print("\n" + "=" * 30)
print("INTERESTING FINDINGS:")
print("=" * 30)
print("1. Setosa species has significantly smaller petal measurements")
print("2. Virginica has the largest measurements on average")
print("3. Versicolor falls between setosa and virginica in most measurements")
print("4. All species have similar sepal width measurements")

# Task 3: Data Visualization
print("\n" + "=" * 50)
print("TASK 3: DATA VISUALIZATION")
print("=" * 50)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')

# 1. Line Chart - Trends across samples (simulating time series)
plt.subplot(2, 2, 1)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.plot(species_data.index[:30], species_data['sepal length (cm)'][:30], 
             marker='o', label=species, alpha=0.7)
plt.title('Sepal Length Trend Across Samples (First 30)')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Bar Chart - Comparison of average measurements across species
plt.subplot(2, 2, 2)
numeric_cols = [col for col in df.columns if col != 'species']
species_means.T.plot(kind='bar', ax=axes[0, 1])
plt.title('Average Measurements by Species')
plt.xlabel('Measurement Type')
plt.ylabel('Average Value (cm)')
plt.xticks(rotation=45)
plt.legend(title='Species')
plt.grid(True, alpha=0.3)

# 3. Histogram - Distribution of sepal length
plt.subplot(2, 2, 3)
df['sepal length (cm)'].hist(bins=15, alpha=0.7, edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add mean line
mean_sepal = df['sepal length (cm)'].mean()
plt.axvline(mean_sepal, color='red', linestyle='--', 
            label=f'Mean: {mean_sepal:.2f}cm')
plt.legend()

# 4. Scatter Plot - Relationship between sepal length and petal length
plt.subplot(2, 2, 4)
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, color in colors.items():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'], 
                alpha=0.7, label=species, c=color)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualizations
print("\nCreating additional visualizations...")

# Box plots for each measurement by species
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    df.boxplot(column=col, by='species')
    plt.title(f'{col} by Species')
    plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.drop('species', axis=1)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Pairplot for comprehensive relationships
sns.pairplot(df, hue='species', diag_kind='hist', palette='husl')
plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
plt.show()

# Final observations
print("\n" + "=" * 50)
print("FINAL OBSERVATIONS AND INSIGHTS")
print("=" * 50)
print("1. Clear separation between setosa and other species in most measurements")
print("2. Strong positive correlation between petal length and petal width")
print("3. Virginica has the largest measurements across all features")
print("4. Setosa has the most distinct characteristics")
print("5. Versicolor serves as an intermediate between setosa and virginica")
print("6. The dataset is well-balanced with 50 samples per species")

print("\nAnalysis completed successfully!")
