import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm

# Load data
original = pd.read_csv("data-challenge-original.csv")
protected = pd.read_csv("protected_data_challenge.csv")

# Save IDs and Names
original_ids = original['Identifier'].values
protected_ids = protected['Identifier'].values
original_names = original['Name'].values
protected_names = protected['Name'].values

# Drop identifiers
original.drop(columns=['Identifier', 'Name'], inplace=True)
protected.drop(columns=['Identifier', 'Name'], inplace=True)

# Tier to number
for df in [original, protected]:
    df["City_Tier"] = df["City_Tier"].str.extract(r'(\d)').astype(int)

# Encode occupations
occ_unique = original['Occupation'].unique()
occ_map = {v: i for i, v in enumerate(occ_unique)}
for df in [original, protected]:
    df["Occupation"] = df["Occupation"].map(occ_map)

# Feature engineering
def engineer(df):
    income = df['Income'] + 1e-5
    df['rent_income_ratio'] = df['Rent'] / income
    df['loan_income_ratio'] = df['Loan_Repayment'] / income
    df['healthcare_ratio'] = df['Healthcare'] / income
    df['non_essential_ratio'] = (df['Entertainment'] + df['Eating_Out'] + df['Miscellaneous']) / income
    df['essential_ratio'] = (df['Groceries'] + df['Insurance'] + df['Utilities']) / income
    burn_cols = ['Rent','Loan_Repayment','Insurance','Groceries','Transport',
                 'Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous']
    df['burn_rate'] = df[burn_cols].sum(axis=1) / income
    return df

original = engineer(original)
protected = engineer(protected)

# Feature list
features = ['Age', 'Occupation', 'City_Tier', 'Dependents', 'Income', 'Healthcare', 'Education',
            'rent_income_ratio', 'loan_income_ratio', 'healthcare_ratio',
            'non_essential_ratio', 'essential_ratio', 'burn_rate']

# Normalize
scaler = MinMaxScaler()
orig_scaled = scaler.fit_transform(original[features])
prot_scaled = scaler.transform(protected[features])

# PCA dimensionality reduction
pca = PCA(n_components=6)  # Adjust components if needed
orig_pca = pca.fit_transform(orig_scaled)
prot_pca = pca.transform(prot_scaled)

# Compute covariance matrix and inverse for Mahalanobis
cov = np.cov(orig_pca.T)
cov_inv = np.linalg.pinv(cov)  # Use pseudo-inverse for stability

# Matching using Mahalanobis Distance
TOP_K = 3
matches = []

for i in tqdm(range(len(prot_pca)), desc="Matching Protected to Original"):
    pvec = prot_pca[i]
    dists = np.array([distance.mahalanobis(pvec, ovec, cov_inv) for ovec in orig_pca])
    top_k_idxs = np.argsort(dists)[:TOP_K]

    for rank, idx in enumerate(top_k_idxs):
        matches.append({
            "Protected_Index": i,
            "Matched_Original_Index": idx,
            "Rank": rank + 1,
            "Distance": round(dists[idx], 4),
            "Protected_ID": protected_ids[i],
            "Original_ID": original_ids[idx],
            "Protected_Name": protected_names[i],
            "Original_Name": original_names[idx]
        })

# Save results
pd.DataFrame(matches).to_csv("matched_results_mahalanobis_topk.csv", index=False)
print("Matching with PCA + Mahalanobis + Top-K complete. Output saved to matched_results_mahalanobis_topk.csv")