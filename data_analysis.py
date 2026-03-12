# data_analysis.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ CSV dosyasını oku
csv_file = "friday.csv"
df = pd.read_csv(csv_file)

# 2️⃣ Veri hakkında bilgi
print("=== VERİ BİLGİSİ ===")
print(df.info())

print("\n=== LABEL DAĞILIMI ===")
print(df['Label'].value_counts())

print("\n=== EKSİK DEĞERLER ===")
print(df.isnull().sum())

# 3️⃣ Sadece BENIGN veriyi al
df_benign = df[df['Label'] == 'BENIGN']
print("\nSadece BENIGN veri sayısı:", len(df_benign))

columns_to_drop = [
    'Src IP dec',
    'Dst IP dec',
    'Timestamp',
    'Label',
    'Attempted Category'
]

columns_to_drop = [col for col in columns_to_drop if col in df_benign.columns]

df_features = df_benign.drop(columns=columns_to_drop)

# 5️⃣ Normalizasyon (0-1 arası)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df_features)

# 6️⃣ 32x32 Image Mapping
num_samples = features_scaled.shape[0]
features_padded = np.zeros((num_samples, 32*32))  # eksikse 0 ile doldur
features_padded[:, :features_scaled.shape[1]] = features_scaled
features_reshaped = features_padded.reshape(num_samples, 1, 32, 32)  # (N,1,32,32)

# 7️⃣ .npy olarak kaydet
np.save("benign_data.npy", features_reshaped)
print("\n✅ benign_data.npy dosyası başarıyla kaydedildi!")