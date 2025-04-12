import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_style('whitegrid')

df = pd.read_excel(r'E:\Python Project 1\Vrinda Store Data Analysis.xlsx')
df.info()
df.rename(columns={'Channel ': 'Channel'}, inplace=True)

def clean_gender(gender):
    if pd.isna(gender):
        return np.nan
    gender = str(gender).strip().lower()
    if gender in ['women', 'w', 'female']:
        return 'Female'
    elif gender in ['men', 'm', 'male']:
        return 'Male'
    else:
        return np.nan

df['Gender'] = df['Gender'].apply(clean_gender)

qty_map = {'one': 1, 'two': 2, 'three': 3, 'One': 1, 'Two': 2, 'Three': 3}
df['Qty'] = df['Qty'].apply(lambda x: qty_map.get(str(x).lower(), x) if isinstance(x, str) else x)
df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

df['Channel'] = df['Channel'].str.strip().str.title().replace('', np.nan)

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', unit='D')
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

def get_age_group(age):
    if pd.isna(age):
        return np.nan
    elif 3 <= age <= 18:
        return 'Teenager'
    elif 19 <= age <= 64:
        return 'Adult'
    elif age >= 65:
        return 'Senior'
    else:
        return 'Other'

df['Age Group'] = df['Age'].apply(get_age_group)

print("Dataset Info After Cleaning:")
df.info()

def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x/1000)}K'
    return f'{int(x)}'


# Objective 1: Total Sales by Ship State (Top 10) 
top_states = df.groupby('ship-state')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x='ship-state', y='Amount', data=top_states, palette='Blues_d')
plt.title('Top 10 Ship States by Sales Amount', fontsize=14)
plt.xlabel('Ship State', fontsize=12)
plt.ylabel('Total Sales (INR)', fontsize=12)
plt.xticks(rotation=45, ha='right')  
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  
plt.tight_layout()
plt.show()

# Objective 2: Sales Distribution by Gender 
sales_by_gender = df.groupby('Gender')['Amount'].sum()
plt.figure(figsize=(8, 8))
plt.pie(sales_by_gender, labels=sales_by_gender.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title('Sales Distribution by Gender', fontsize=14)
plt.tight_layout()
plt.show()

# Objective 3: Category-wise Sales Distribution 
sales_by_category = df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Amount', data=sales_by_category, palette='Greens_d')
plt.title('Sales Distribution by Product Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales (INR)', fontsize=12)
plt.xticks(rotation=45, ha='right') 
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  
plt.tight_layout()
plt.show()

# Objective 4: Channel-wise Order Volume 
plt.figure(figsize=(10, 6))
sns.countplot(y='Channel', data=df, order=df['Channel'].value_counts().index, palette='Purples_d')
plt.title('Order Volume by Sales Channel', fontsize=14)
plt.xlabel('Number of Orders', fontsize=12)
plt.ylabel('Channel', fontsize=12)
plt.tight_layout()
plt.show()

# Objective 5: Order Status Breakdown 
status_counts = df['Status'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title('Order Status Breakdown', fontsize=14)
plt.tight_layout()
plt.show()

print("Data cleaning and visualizations completed successfully.")