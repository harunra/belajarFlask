import pandas as pd  # Library untuk manipulasi data dalam bentuk DataFrame
import pickle  # Library untuk memuat model machine learning yang sudah disimpan
from flask import Flask, request, jsonify  # Flask untuk membuat web API, request untuk menerima data, jsonify untuk mengirim response JSON
from flask_cors import CORS  # Import Flask-CORS untuk mengizinkan akses dari domain berbeda (Cross-Origin Resource Sharing)
import os


# Membuat aplikasi Flask
app = Flask(__name__)


# Aktifkan CORS - Mengizinkan semua domain untuk mengakses API ini
# CORS diperlukan agar frontend (seperti website) dari domain lain bisa mengakses API ini
CORS(app)  # Mengizinkan semua domain


# Set working directory to the belajar folder
belajar_dir = os.path.dirname(os.path.abspath(__file__))


# Muat model yang sudah disimpan dari file model.pkl
# Model ini adalah hasil training yang sudah dilakukan sebelumnya
model_path = os.path.join(belajar_dir, '[Text] Diabetes', 'model.pkl')
try:
    with open(model_path, 'rb') as file:  # 'rb' = read binary, karena file pickle adalah file binary
        model = pickle.load(file)  # Memuat model ke dalam variabel 'model'
    print(f"✓ Model loaded from {model_path}")
except FileNotFoundError:
    print(f"✗ Model file not found: {model_path}")
    model = None


# Muat scaler yang sudah disimpan untuk normalisasi data input
# Scaler ini penting karena model dilatih dengan data yang sudah dinormalisasi
scaler_path = os.path.join(belajar_dir, '[Text] Diabetes', 'scaler.pkl')
try:
    with open(scaler_path, 'rb') as file:  # Membuka file scaler.pkl
        scaler = pickle.load(file)  # Memuat scaler ke dalam variabel 'scaler'
    print(f"✓ Scaler loaded from {scaler_path}")
except FileNotFoundError:
    print(f"✗ Scaler file not found: {scaler_path}")
    scaler = None


# Endpoint untuk halaman home /
# Endpoint ini hanya menampilkan pesan selamat datang ketika user mengakses root URL
@app.route('/')
def welcome():
    return jsonify({
        'message': 'Selamat Datang di API Diabetes Prediction Model',
        'version': '1.0',
        'endpoints': {
            'predict': 'POST /predict - Untuk memprediksi diabetes'
        }
    })


# Endpoint untuk memprediksi diabetes
# Endpoint ini menerima data pasien dan mengembalikan prediksi diabetes
# Method POST digunakan karena kita mengirim data untuk diproses
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Cek apakah model dan scaler sudah berhasil dimuat
        if model is None or scaler is None:
            return jsonify({'error': 'Model atau Scaler tidak tersedia'}), 500

        # Ambil data dari request body dalam format JSON
        # Data dikirim dari client (frontend/Postman) dalam format JSON
        data = request.get_json()

        # Validasi bahwa data tidak None
        if not data:
            return jsonify({'error': 'Request body tidak boleh kosong'}), 400

        # Validasi bahwa semua field diperlukan ada
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Field yang hilang: {", ".join(missing_fields)}'}), 400

        # Buat DataFrame dengan nama kolom yang sesuai
        # DataFrame diperlukan karena model sklearn membutuhkan input dalam format DataFrame
        # Kolom-kolom ini harus sesuai dengan kolom yang digunakan saat training model
        input_data = pd.DataFrame([{
            "Pregnancies": float(data['Pregnancies']),  # Jumlah kehamilan
            "Glucose": float(data['Glucose']),  # Kadar glukosa dalam darah
            "BloodPressure": float(data['BloodPressure']),  # Tekanan darah
            "SkinThickness": float(data['SkinThickness']),  # Ketebalan kulit
            "Insulin": float(data['Insulin']),  # Kadar insulin
            "BMI": float(data['BMI']),  # Body Mass Index (indeks massa tubuh)
            "DiabetesPedigreeFunction": float(data['DiabetesPedigreeFunction']),  # Fungsi silsilah diabetes
            "Age": float(data['Age'])  # Umur pasien
        }])

        # Normalisasi data input menggunakan scaler yang sama saat training
        # Ini SANGAT PENTING karena model dilatih dengan data yang sudah dinormalisasi
        # Jika tidak dinormalisasi, prediksi akan salah
        input_data_scaled = scaler.transform(input_data)

        # Melakukan prediksi dengan model yang sudah dimuat
        # predict() mengembalikan array dengan nilai 0 (negatif) atau 1 (positif diabetes)
        prediction = model.predict(input_data_scaled)

        # Mendapatkan probabilitas prediksi untuk setiap kelas
        # predict_proba() mengembalikan probabilitas untuk kelas 0 dan kelas 1
        probabilities = model.predict_proba(input_data_scaled)

        # Probabilitas positif dan negatif dalam bentuk persentase
        probability_negative = probabilities[0][0] * 100  # Probabilitas untuk kelas 0 (tidak diabetes) - dikali 100 untuk persentase
        probability_positive = probabilities[0][1] * 100  # Probabilitas untuk kelas 1 (diabetes) - dikali 100 untuk persentase

        # Prediksi output (0 atau 1, di mana 1 berarti positif diabetes)
        # Membuat pesan yang mudah dipahami berdasarkan hasil prediksi
        if prediction[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. Kemungkinan menderita diabetes adalah {probability_positive:.2f}%.'
            risk_level = 'HIGH'
        else:
            result = 'Hasil prediksi menunjukkan Anda kemungkinan rendah terkena diabetes.'
            risk_level = 'LOW'

        # Kembalikan hasil prediksi dan probabilitas dalam bentuk JSON
        # jsonify() mengubah dictionary Python menjadi response JSON yang bisa dibaca oleh client
        return jsonify({
            'success': True,
            'prediction': result,
            'risk_level': risk_level,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",  # Format 2 desimal untuk probabilitas negatif
                'positive': f"{probability_positive:.2f}%"  # Format 2 desimal untuk probabilitas positif
            }
        }), 200

    except ValueError as e:
        # Jika ada error konversi tipe data (misalnya string yang tidak bisa jadi float)
        return jsonify({'error': f'Error konversi data: {str(e)}'}), 400
    except Exception as e:
        # Jika terjadi error lainnya (misalnya data tidak lengkap atau format salah)
        # Kembalikan pesan error dengan status code 400 (Bad Request)
        return jsonify({'error': str(e)}), 400


# Jalankan aplikasi Flask
# if __name__ == '__main__': memastikan kode ini hanya dijalankan jika file ini dieksekusi langsung
# (tidak dijalankan jika file ini diimpor sebagai module)
if __name__ == '__main__':
    app.run(debug=True)  # debug=True memberikan informasi error yang detail dan auto-reload saat development
