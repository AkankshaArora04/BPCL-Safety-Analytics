import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────
df = pd.read_excel('data/VA ALERTS.XLSX')

# ── Data Prep ──────────────────────────────────────────
df['created'] = pd.to_datetime(df['created'])
df['month'] = df['created'].dt.to_period('M')
df['year'] = df['created'].dt.year

print("Data loaded! Shape:", df.shape)
print("\nTop violations:")
print(df['interlockname'].value_counts().head(10))

# ── Plot 1: Top 10 Violations ──────────────────────────
plt.figure(figsize=(12, 6))
top_violations = df['interlockname'].value_counts().head(10)
sns.barplot(x=top_violations.values, y=top_violations.index, palette='Reds_r')
plt.title('Top 10 Safety Violations at BPCL', fontsize=14, fontweight='bold')
plt.xlabel('Number of Alerts')
plt.ylabel('Violation Type')
plt.tight_layout()
plt.savefig('plot1_top_violations.png')
plt.show()
print("Plot 1 saved!")

# ── Plot 2: Violations per Unit ────────────────────────
plt.figure(figsize=(12, 6))
unit_counts = df['unitName'].value_counts().head(15)
sns.barplot(x=unit_counts.values, y=unit_counts.index, palette='Blues_r')
plt.title('Violations by BPCL Unit', fontsize=14, fontweight='bold')
plt.xlabel('Number of Alerts')
plt.ylabel('Unit Name')
plt.tight_layout()
plt.savefig('plot2_violations_by_unit.png')
plt.show()
print("Plot 2 saved!")

# ── Plot 3: Violations Over Time ───────────────────────
plt.figure(figsize=(14, 5))
monthly = df.groupby('month').size()
monthly.plot(kind='line', marker='o', color='red', linewidth=2)
plt.title('Safety Violations Over Time (Monthly)', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Violations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_violations_over_time.png')
plt.show()
print("Plot 3 saved!")

# ── Plot 4: Helmet vs Other Violations ────────────────
df['is_helmet'] = df['interlockname'].str.contains('Helmet', case=False, na=False)
helmet_counts = df['is_helmet'].value_counts()
helmet_counts.index = ['Other Violations', 'Helmet Violations']

plt.figure(figsize=(7, 7))
plt.pie(helmet_counts.values, labels=helmet_counts.index,
        autopct='%1.1f%%', colors=['#4C72B0', '#DD4444'],
        startangle=90, textprops={'fontsize': 13})
plt.title('Helmet vs Other Violations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_helmet_pie.png')
plt.show()
print("Plot 4 saved!")

print("\n✅ All 4 plots saved in your project folder!")

# ── ML Model: Predict if violation is Helmet related ──
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("\n🤖 Building ML Model...")

# Features banao
df['hour'] = df['created'].dt.hour
df['dayofweek'] = df['created'].dt.dayofweek
df['is_helmet'] = df['interlockname'].str.contains('Helmet', case=False, na=False).astype(int)

# Unit name ko number mein convert karo
le = LabelEncoder()
df['unit_encoded'] = le.fit_transform(df['unitName'])

# X = input features, y = output label
X = df[['hour', 'dayofweek', 'sapId', 'unit_encoded']]
y = df['is_helmet']

# Train/Test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train karo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict karo
y_pred = model.predict(X_test)

# Results print karo
print("\n✅ Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=['Other', 'Helmet']))

# Feature Importance Graph
plt.figure(figsize=(8, 5))
feat_imp = pd.Series(model.feature_importances_, index=['Hour', 'Day of Week', 'SAP ID', 'Unit'])
feat_imp.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance — What predicts Helmet Violations?', fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot5_feature_importance.png')
plt.show()
print("Plot 5 saved!")

print("\n🎉 Project 60% Done!")