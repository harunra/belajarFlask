import numpy as np  # Mengimpor library NumPy untuk manipulasi data numerik.
import pandas as pd  # Mengimpor library Pandas untuk menangani data dalam format DataFrame.
import pickle  # Mengimpor library Pickle untuk menyimpan dan memuat model yang telah dilatih.
from sklearn.model_selection import train_test_split  # Mengimpor fungsi train_test_split dari scikit-learn untuk membagi data menjadi data pelatihan dan pengujian.
from sklearn.neighbors import KNeighborsClassifier  # Mengimpor algoritma K-Nearest Neighbors (KNN) dari scikit-learn untuk model klasifikasi.
import os

# Set working directory to the belajar folder
belajar_dir = os.path.dirname(os.path.abspath(__file__))

def load_diabetes_data(filepath):
    """Load diabetes dataset from CSV file."""
    data = pd.read_csv(filepath)
    return data

def train_diabetes_model(data):
    """Train KNN model on diabetes dataset."""
    X = data.drop('Outcome', axis=1)  # Membuat data prediktor (fitur) dengan menghapus kolom 'Outcome' dari dataset.
    y = data['Outcome']  # Menentukan variabel target (klasifikasi) yang akan digunakan untuk model.
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    score = model.score(X_test, y_test)
    return score

def save_model(model, filepath):
    """Save trained model to disk."""
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model from disk."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filepath}")
    return model

def predict_diabetes(model, input_data):
    """Make prediction on new data."""
    prediction = model.predict([input_data])
    return prediction[0]

def main():
    """Main function to run the diabetes ML pipeline."""
    print("=== Diabetes ML Model Pipeline ===\n")
    
    # File paths
    csv_path = os.path.join(belajar_dir, '[Text] Diabetes', 'diabetes.csv')
    model_path = os.path.join(belajar_dir, '[Text] Diabetes', 'model.pkl')
    
    # Load data
    print("Loading diabetes dataset...")
    data = load_diabetes_data(csv_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns[:-1])}\n")
    
    # Train model
    print("Training KNN model...")
    model, X_train, X_test, y_train, y_test = train_diabetes_model(data)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")
    
    # Evaluate model
    print("Evaluating model...")
    score = evaluate_model(model, X_test, y_test)
    print(f"Model score (accuracy): {score:.4f}\n")
    
    # Save model
    print("Saving model...")
    save_model(model, model_path)
    
    # Save test data
    X_test.to_csv(os.path.join(belajar_dir, '[Text] Diabetes', 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(belajar_dir, '[Text] Diabetes', 'y_test.csv'), index=False)
    print("Test data saved.\n")
    
    # Example prediction
    print("=== Example Prediction ===")
    sample_data = X_test.iloc[0].values  # Take first test sample
    prediction = predict_diabetes(model, sample_data)
    actual = y_test.iloc[0]
    print(f"Sample input: {sample_data}")
    print(f"Prediction: {prediction}")
    print(f"Actual: {actual}")
    print(f"Correct: {prediction == actual}\n")
    
    return model

if __name__ == '__main__':
    model = main()




















# import numpy as np  # Mengimpor library NumPy untuk manipulasi data numerik.
# import pandas as pd  # Mengimpor library Pandas untuk menangani data dalam format DataFrame.
# import pickle  # Mengimpor library Pickle untuk menyimpan dan memuat model yang telah dilatih.
# from sklearn.model_selection import train_test_split  # Mengimpor fungsi train_test_split dari scikit-learn untuk membagi data menjadi data pelatihan dan pengujian.
# from sklearn.neighbors import KNeighborsClassifier  # Mengimpor algoritma K-Nearest Neighbors (KNN) dari scikit-learn untuk model klasifikasi.

# # Load data  # Memuat dataset diabetes dari file CSV yang berisi informasi pasien dan hasil pemeriksaan diabetes.
# data = pd.read_csv('diabetes.csv')  # Membaca data dari file 'diabetes.csv' menggunakan pandas.

# X = data.drop('Outcome', axis=1)  # Membuat data prediktor (fitur) dengan menghapus kolom 'Outcome' dari dataset.
# y = data['Outcome']  # Menentukan variabel target (klasifikasi) yang akan digunakan untuk model.

# # Split data  # Memisahkan data menjadi dataset pelatihan (80%) dan pengujian (20%).
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Menggunakan fungsi train_test_split dari scikit-learn untuk membagi data.

# # Train model  # Memulai proses pelatihan model dengan menggunakan data pelatihan (X_train, y_train).
# model = KNeighborsClassifier()  # Membuat objek model K-Nearest Neighbors (KNN).
# model.fit(X_train, y_train)  # Melatih model menggunakan dataset pelatihan.

# # Evaluate model  # Menilai performa model dengan menggunakan data pengujian (X_test, y_test).
# score = model.score(X_test, y_test)  # Menghitung akurasi model menggunakan dataset pengujian.
# print(f'Model score: {score}')  # Menampilkan skor akurasi model.

# # Save model  # Menyimpan model yang telah dilatih untuk keperluan penggunaan kembali di kemudian hari.
# with open('model.pkl', 'wb') as file:  # Membuka file 'model.pkl' dalam mode write-byte.
#     pickle.dump(model, file)  # Menyimpan model menggunakan format Pickle.

# # Save test data  # Menyimpan data hasil pengujian (X_test dan y_test) untuk referensi atau analisis lebih lanjut.
# X_test.to_csv('X_test.csv', index=False)  # Menyimpan data X_test ke dalam file CSV.
# y_test.to_csv('y_test.csv', index=False)  # Menyimpan data y_test ke dalam file CSV.



