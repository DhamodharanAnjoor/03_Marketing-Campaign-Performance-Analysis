import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing      import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score, f1_score


print("MARKETING CAMPAIGN")


# ── STEP 1: Merge datasets
print(" Step 1 — Loading 3 datasets...")

nykaa   = pd.read_csv('nykaa_campaign_data_with_nulls.csv')
purplle = pd.read_csv('purplle_campaign_data_with_nulls.csv')
tira    = pd.read_csv('tira_campaign_data_with_nulls.csv')

nykaa['Campaign_Name']   = 'Nykaa'
purplle['Campaign_Name'] = 'Purplle'
tira['Campaign_Name']    = 'Tira'

df = pd.concat([nykaa, purplle, tira], ignore_index=True)
df.drop(columns=['Campaign_ID'], inplace=True)
print(f"Merged shape : {df.shape}")

# ── STEP 2: CLEAN NULLS 
print("Step 2 — Cleaning null values...")

cat_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Language', 'Customer_Segment', 'Campaign_Name']
num_cols = ['Duration', 'Impressions', 'Clicks', 'Leads',
            'Conversions', 'Revenue', 'Acquisition_Cost',
            'ROI', 'Engagement_Score']

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['Date'] = df['Date'].ffill()
df[['Duration','Impressions','Clicks','Leads','Conversions']] = \
    df[['Duration','Impressions','Clicks','Leads','Conversions']].astype(int)

print(f"Nulls remaining : {df.isnull().sum().sum()}")

# ── STEP 3: FEATURE ENGINEERING 
print("Step 3 — Feature engineering...")

# Target column: 1 = Profit, 0 = Loss
df['Profit_Flag'] = (df['ROI'] > 0).astype(int)

# Channel_Used → 6 binary columns (Email, Facebook, etc.)
mlb      = MultiLabelBinarizer()
ch_lists = df['Channel_Used'].apply(
    lambda x: [c.strip() for c in str(x).split(',')]
)
ch_df = pd.DataFrame(
    mlb.fit_transform(ch_lists),
    columns=mlb.classes_, index=df.index
)
df = pd.concat([df, ch_df], axis=1)
df.drop(columns=['Channel_Used', 'Date'], inplace=True)

# Encode text columns → numbers
le = LabelEncoder()
for col in ['Campaign_Name','Campaign_Type','Target_Audience',
            'Language','Customer_Segment']:
    df[col] = le.fit_transform(df[col].astype(str))

df.to_csv('clean_campaign_data.csv', index=False)
print(f"    clean_campaign_data.csv saved  {df.shape}")

# ── STEP 4: BASIC EDA 
print("Step 4 — Basic EDA summary...")
print(f"   Total campaigns : {len(df):,}")
print(f"   Profit (1)      : {(df['Profit_Flag']==1).sum():,}  ({df['Profit_Flag'].mean()*100:.1f}%)")
print(f"   Loss   (0)      : {(df['Profit_Flag']==0).sum():,}  ({(1-df['Profit_Flag'].mean())*100:.1f}%)")
print(f"   Avg Revenue     : ₹{df['Revenue'].mean():,.0f}")
print(f"   Avg ROI         : {df['ROI'].mean():.3f}")
print(f"   Avg Engagement  : {df['Engagement_Score'].mean():.2f}")

# ── STEP 5: TRAIN MODEL 
print("\nStep 5 — Training classification model...")

FEATURES = [
    'Campaign_Name', 'Campaign_Type', 'Target_Audience', 'Language',
    'Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions',
    'Revenue', 'Acquisition_Cost', 'Engagement_Score',
    'Email', 'Facebook', 'Google', 'Instagram', 'WhatsApp', 'YouTube'
]

X = df[FEATURES]
y = df['Profit_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"    Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"    F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")

# ── STEP 6: SAVE PKL ─────────────────────────────────────
with open('profit_loss_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\n')
print("profit_loss_classifier.pkl saved!")

