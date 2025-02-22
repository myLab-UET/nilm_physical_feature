import pandas as pd
import matplotlib.pyplot as plt
from nilm_dao import *
import umap

iawe_df = get_iawe_data("test", True).to_pandas()
lb = get_label_encoder("iawe")
full_features_df = iawe_df[["Irms", "P", "MeanPF", "S", "Q", "Label"]]
full_features_df["Label"] = lb.inverse_transform(full_features_df["Label"])
full_features_df = full_features_df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), 3000), random_state=42)).reset_index(drop=True)
print(len(full_features_df))
print(full_features_df.head())

dict_name = {
    "ac1_s1": "Air Cond. 1 S1",
    "ac1_s2": "Air Cond. 1 S2",
    "ac1_s3": "Air Cond. 1 S3",
    "ac2_s1": "Air Cond. 2 S1",
    "ac2_s2": "Air Cond. 2 S2",
    "ac2_s3": "Air Cond. 2 S3",
    "clothes_iron": "Clothes Iron",
    "computer": "Computer",
    "fridges_s1": "Fridge",
    "television": "Television",
    "washing_machine_s1": "Washing Machine",
    "wet_appliance_s1": "Water purifier"
}

full_features_df = full_features_df.replace({"Label": dict_name})
p_data = full_features_df[["P", "Label"]]

# Plt settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 6))
reducer = umap.UMAP()
embedding = reducer.fit_transform(p_data[["P"]].values, verbose=True)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(p_data["Label"])[0], cmap="Spectral", s=5)
plt.gca().set_aspect("equal", "datalim")
# plt.title("UMAP Projection of the IAWE dataset (Only Real power)", fontsize=16)
plt.legend(handles=scatter.legend_elements()[0], labels=list(p_data["Label"].unique()), bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results/iawe_P_umap.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
reducer = umap.UMAP()
embedding = reducer.fit_transform(full_features_df[["Irms", "P", "MeanPF", "S", "Q"]].values, verbose=True)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(full_features_df["Label"])[0], cmap="Spectral", s=5)
plt.gca().set_aspect("equal", "datalim")
# plt.title("UMAP Projection of the IAWE dataset (All Features)", fontsize=16)
plt.legend(handles=scatter.legend_elements()[0], labels=list(full_features_df["Label"].unique()), bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results/iawe_full_features_umap.png", bbox_inches='tight')
plt.close()