import pandas as pd


# Split train.csv into train_lateral, train_frontal_ap, and train_frontal_pa csvs.
def split_training_views():
    # What other read_csv parameters do we need?
    label_path = "CheXpert-v1.0-small/train.csv"
    label_df = pd.read_csv(label_path)
    # Filter training labels into three different view types.
    lateral = label_df[label_df['Frontal/Lateral'] == 'Lateral']
    frontal_ap = label_df[label_df['AP/PA'] == 'AP']
    frontal_pa = label_df[label_df['AP/PA'] == 'PA']
    # write the split dfs into separate csvs to load with CheXpert class.
    lateral.to_csv("train_lateral.csv", index=False)
    frontal_ap.to_csv("train_frontal_ap.csv", index=False)
    frontal_pa.to_csv("train_frontal_pa.csv", index=False)


# Split train.csv into train_lateral, train_frontal_ap, and train_frontal_pa csvs.
split_training_views()
