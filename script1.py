import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm

def load_data(original_path, protected_path):
    print("📊 Loading data...")
    original = pd.read_csv(original_path)
    protected = pd.read_csv(protected_path)
    
    # Save IDs and Names for matching between protected and original given
    original_ids = original['Identifier'].values
    protected_ids = protected['Identifier'].values
    original_names = original['Name'].values
    protected_names = protected['Name'].values

    # Drop identifiers cause they are bad?! for matching maybe this approach changes in future
    original.drop(columns=['Identifier', 'Name'], inplace=True)
    protected.drop(columns=['Identifier', 'Name'], inplace=True)
    
    return original, protected, original_ids, protected_ids, original_names, protected_names

def preprocess_data(original, protected):
    print("🔄 Preprocessing data...")
    # Tier to number obviously so that we can use it in PCA maths
    for df in [original, protected]:
        df["City_Tier"] = df["City_Tier"].str.extract(r'(\d)').astype(int)

    # Encode occupations, this part of the code optimized by llm
    occ_unique = original['Occupation'].unique()
    occ_map = {v: i for i, v in enumerate(occ_unique)}
    for df in [original, protected]:
        df["Occupation"] = df["Occupation"].map(occ_map)
        
    return original, protected


def engineer_features(df):
    #obciously mathematical operations, like dividing by income
    # to get ratios
    # Add small value to avoid division by zero
    # This is a common practice to avoid division by zero errors
    # and to ensure numerical stability
    # when dealing with financial data
    # where income can be very small
    # or even zero in some cases
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


def apply_feature_engineering(original, protected):
    #name says it all, and yes a debug stmt
    print("Engineering features...")
    original = engineer_features(original)
    protected = engineer_features(protected)
    return original, protected


def normalize_and_reduce_dimensions(original, protected, n_components=6):
    # Normalize and reduce dimensions using PCA, debug stmt too
    print("📉 Normalizing and reducing dimensions...")
    features = ['Age', 'Occupation', 'City_Tier', 'Dependents', 'Income', 'Healthcare', 'Education',
                'rent_income_ratio', 'loan_income_ratio', 'healthcare_ratio',
                'non_essential_ratio', 'essential_ratio', 'burn_rate']
    
    # Normalize
    scaler = MinMaxScaler()
    orig_scaled = scaler.fit_transform(original[features])
    prot_scaled = scaler.transform(protected[features])
    
    # PCA dimensionality reduction
    pca = PCA(n_components=n_components)
    orig_pca = pca.fit_transform(orig_scaled)
    prot_pca = pca.transform(prot_scaled)
    
    return orig_pca, prot_pca, features, scaler, pca


def match_using_mahalanobis(orig_pca, prot_pca, protected_ids, original_ids, protected_names, original_names, top_k=3):
    
    print("🔍 Matching using Mahalanobis distance...")
    # Compute covariance matrix and inverse for Mahalanobis
    cov = np.cov(orig_pca.T)
    cov_inv = np.linalg.pinv(cov)  # Use pseudo-inverse for stability
    
    matches = []
    for i in tqdm(range(len(prot_pca)), desc="Matching Protected to Original"):
        pvec = prot_pca[i]
        dists = np.array([distance.mahalanobis(pvec, ovec, cov_inv) for ovec in orig_pca])
        top_k_idxs = np.argsort(dists)[:top_k]
    
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
            
    return matches


def save_results(matches, output_path):
    print(f"Saving results to {output_path}...")
    pd.DataFrame(matches).to_csv(output_path, index=False)
    print(f"Matching complete. Output saved to {output_path}")


def main():
    """Main function to coordinate the workflow."""
    # Configuration
    original_path = "data-challenge-original.csv"
    protected_path = "protected_data_challenge.csv"
    output_path = "matched_results_mahalanobis_topk1.csv"
    top_k = 3
    pca_components = 6
    
    # Execute workflow
    original, protected, original_ids, protected_ids, original_names, protected_names = load_data(
        original_path, protected_path
    )
    
    original, protected = preprocess_data(original, protected)
    
    original, protected = apply_feature_engineering(original, protected)
    
    orig_pca, prot_pca, features, scaler, pca = normalize_and_reduce_dimensions(
        original, protected, n_components=pca_components
    )
    
    matches = match_using_mahalanobis(
        orig_pca, prot_pca, protected_ids, original_ids, protected_names, original_names, top_k=top_k
    )
    
    save_results(matches, output_path)

if __name__ == "__main__":
    main()