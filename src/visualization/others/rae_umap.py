import pandas as pd
import matplotlib.pyplot as plt
from nilm_dao import *
import umap

rae_df = get_rae_data("test", False).to_pandas()
lb = get_label_encoder("rae")
full_features_df = rae_df[["Irms", "P", "MeanPF", "S", "Q", "Label"]]
full_features_df["Label"] = lb.inverse_transform(full_features_df["Label"])
full_features_df = full_features_df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), 3000), random_state=42)).reset_index(drop=True)
print(len(full_features_df))
print(full_features_df.head())
print(full_features_df["Label"].unique())
dict_name = {
    'basement_blue_plugs': "Basement Blue Plugs",
    'bathrooms': "Bathrooms",
    'clothes_dryer_s1': 'Clothes dryer S1', 
    'clothes_dryer_s2': 'Clothes dryer S2',
    'clothes_washer': 'Clothes washer',
    'fridge_s1': 'Fridge S1',
    'fridge_s2': 'Fridge S2',
    'furnace_and_hot_water_unit': "Furnace and Hot Water Unit",
    'garage_sub_panel': 'Garage Sub Panel',
    'heat_pump': 'Heat Pump',
    'home_office': 'Home Office',
    'kitchen_counter_plugs': 'Kitchen Plugs',
    'kitchen_dishwasher': "Diswasher",
    'kitchen_oven_s1': 'Oven S1',
    'kitchen_oven_s2': 'Oven S2',
    'kitchen_oven_s3': 'Oven S3',
    'lp16_s1': 'L&P 16 S1',
    'lp20_s2': 'L&P 20 S2', 
    'lp320_s1': 'L&P 320 S1',
    'lp3_s2': 'L&P 3 S2',
    'misc_plugs': 'Miscellaneous Plugs',
    'rental_suite_sub_panel': 'Rental Suite Sub Panel',
    'upstairs_bedroom_AFCI_arc-fault_plugs': 'Bedroom AFCI',
    'upstairs_plug_and_lights': 'Upstairs P&L',
}

full_features_df = full_features_df.replace({"Label": dict_name})
p_data = full_features_df[["P", "Label"]]
unique_labels = list(full_features_df["Label"].unique())

# Plt settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 16})


print("Starting UMAP projection...")
plt.figure(figsize=(13, 7))  # Reduced width from 18 to 12
reducer = umap.UMAP(min_dist=0.1)
embedding = reducer.fit_transform(full_features_df[["Irms", "P", "MeanPF", "S", "Q"]].values, verbose=True)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(full_features_df["Label"])[0], cmap="Spectral", s=2)
plt.gca().set_aspect("equal", "datalim")
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Spectral(i/len(unique_labels)), 
                            label=unique_labels[i], markersize=10) for i in range(len(unique_labels))]
plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=4)
plt.tight_layout()
plt.savefig("results/rae_full_features_umap.png", bbox_inches='tight')
plt.close()
print("Full features done!")

plt.figure(figsize=(13, 7))  # Reduced width from 18 to 12
reducer = umap.UMAP(min_dist=0.1)
embedding = reducer.fit_transform(p_data[["P"]].values, verbose=True)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(p_data["Label"])[0], cmap="Spectral", s=2)
plt.gca().set_aspect("equal", "datalim")
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Spectral(i/len(unique_labels)), 
                              label=unique_labels[i], markersize=10) for i in range(len(unique_labels))]
plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=4)
plt.tight_layout()
plt.savefig("results/rae_P_umap.png", bbox_inches='tight')
plt.close()
print("P data done!")

