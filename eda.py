import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

print("Dataset loaded successfully")  
print("\nShape of dataset:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())


print("\nSummary statistics:")  
print(df.describe())


print("\nMissing values per column:") 
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())



plt.figure(figsize=(6,4))  
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.boxplot(x=df['Amount'])
plt.title("Boxplot of Transaction Amount")
plt.xlabel("Amount")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


print("\nCorrelation of features with target variable (Class):")
print(df.corr()['Class'].sort_values(ascending=False))


print("\nClass percentage distribution:")
print(df['Class'].value_counts(normalize=True) * 100)