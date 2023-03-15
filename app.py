import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy import signal
import scipy
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier


st.set_option('deprecation.showPyplotGlobalUse', False)

# count amount files in folder and return list of files

def count_files(path):
    import os
    files = os.listdir(path)
    return files

# make dataframe column file and label
df_major = pd.DataFrame(columns=['file', 'label'])
df_minor = pd.DataFrame(columns=['file', 'label'])

df_major['file'] = count_files('major_minor_audio_dataset/Major')
df_major['label'] = 'major'

df_minor['file'] = count_files('major_minor_audio_dataset/Minor')
df_minor['label'] = 'minor'

df = pd.concat([df_major, df_minor], ignore_index=True)


st.title('Major and Minor Audio Dataset')
st.subheader('DF 10 first rows')
st.table(df.head(10))
st.subheader('DF 10 last rows')
st.table(df.tail(10))
st.write("Jumlah Data : ", df.shape[0])

st.write("Jumlah Data Untuk Kelas Major : ", len(df_major))
st.write("Jumlah Data Untuk Kelas Minor : ", len(df_minor))

st.subheader('Data Checking')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset/Major/Major_0.wav')
sample_audio_array = np.array(sample_audio)

# Sampel Rate adalah jumlah data yang diambil dalam satu detik
st.write("Sample audio file : major_minor_audio_dataset/Major/Major_0.wav")
# Panjang array audio
st.write("Panjang Data Audio : ", len(sample_audio_array))
st.write("Sample Rate : ", sample_rate)
# Verifikasi Detik Audio
st.write("Durasi Audio : ", len(sample_audio_array)/sample_rate, "detik")


def apply_lowpass_filter(audio, sr, cutoff):
    # Membuat filter Butterworth orde 6
    b, a = signal.butter(6, cutoff, btype='lowpass', fs=sr)
    # Menerapkan filter pada audio
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio

# for i in range(len(df)):
#     audio, sr = sf.read('major_minor_audio_dataset/'+df['label'][i]+'/'+df['file'][i])
#     filtered_audio = apply_lowpass_filter(audio, sr, 1000)
#     sf.write('major_minor_audio_dataset_noise_reduction/'+df['label'][i]+'/'+df['file'][i], filtered_audio, sr)

st.title('Data Preprocessing')
st.subheader('Before Noise Reduction')
st.audio('major_minor_audio_dataset/Major/Major_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset/Major/Major_0.wav')
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array[0:100])
plt.title('Original Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()



st.subheader("After Noise Reduction (Lowpass Filter) (Cutoff = 1000 Hz)")
st.audio('major_minor_audio_dataset_noise_reduction/Major/Major_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset_noise_reduction/Major/Major_0.wav')
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array[0:100])
plt.title('Filtered Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()

st.subheader("Normalization")

# for i in range(len(df)):
#     audio, sr = sf.read('major_minor_audio_dataset_noise_reduction/'+df['label'][i]+'/'+df['file'][i])
#     # Hitung nilai RMS audio
#     rms = np.sqrt(np.mean(np.square(audio)))
#     # Hitung gain untuk mencapai target dB
#     target_amp = 10**(-12/20)
#     gain = target_amp / rms
#     # Normalisasi audio
#     normalized_audio = audio * gain
#     sf.write('major_minor_audio_dataset_normalized/'+df['label'][i]+'/'+df['file'][i], normalized_audio, sr)

    
st.subheader("Before Normalization")
st.audio('major_minor_audio_dataset_noise_reduction/Major/Major_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset_noise_reduction/Major/Major_0.wav')
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array)
plt.title('Original Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()

st.subheader("Before Normalization")
st.audio('major_minor_audio_dataset_noise_reduction/Minor/Minor_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset_noise_reduction/Minor/Minor_0.wav')
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array)
plt.title('Original Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()

st.subheader("After Normalization Sample File Major_0 (Target dB = -12 dB)")
st.audio('major_minor_audio_dataset_normalized/Major/Major_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset_normalized/Major/Major_0.wav') 
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array)
plt.title('Normalized Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()

st.subheader("After Normalization Sample File Minor_0 (Target dB = -12 dB)")
st.audio('major_minor_audio_dataset_normalized/Minor/Minor_0.wav')
sample_audio, sample_rate = sf.read('major_minor_audio_dataset_normalized/Minor/Minor_0.wav')
sample_audio_array = np.array(sample_audio)
plt.plot(sample_audio_array)
plt.title('Normalized Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
st.pyplot()


st.header("Segmentasi Audio Menggunakan Frame Blocking")
st.write("Frame Length : 1024 dan Hop Length : 512")
st.write("Frame Blocking pada Sample File Major_0 dapat diartikan adanya pembagian data audio menjadi beberapa bagian dengan panjang 1024 data dan setiap bagian tersebut dipindahkan sebanyak 512 data ke kanan.")

sample_audio, sample_rate = sf.read('major_minor_audio_dataset_normalized/Major/Major_0.wav')
sample_audio_array = np.array(sample_audio)

frame_length = 1024
hop_length = 512

num_frames = 1 + int((len(sample_audio_array) - frame_length) / hop_length)

frames = []
fft_frames = []
for i in range(num_frames):
    frame = sample_audio_array[i * hop_length : i * hop_length + frame_length]
    fft_frame = np.fft.fft(frame)
    # convert from complex number to real number
    fft_frame = np.real(np.abs(fft_frame))
    fft_frames.append(fft_frame)
    frames.append(frame)

# i want to show sample two row of frames
df_sample_frame_blocking = pd.DataFrame(frames)
df_sample_frame_blocking = df_sample_frame_blocking.iloc[0:4, 0:1024]
st.write("Data dibawah ini merupakan contoh frame blocking pada sample file Major_0 yang terdiri dari 4 baris dan 1024 kolom serta setiap perpindahan data sebanyak 512 data ke kanan.")
st.write(df_sample_frame_blocking)

df_sample_frame_blocking_fft = pd.DataFrame(fft_frames)
df_sample_frame_blocking_fft = df_sample_frame_blocking_fft.iloc[0:4, 0:1024]
st.write("Data Dibawah ini merupakan contoh hasil konversi dari frame blocking menjadi bentuk frekuensi atau FFT (Fast Fourier Transform).")
st.write(df_sample_frame_blocking_fft)


frames = np.array(frames)
fft_frames = np.array(fft_frames)

plt.figure(figsize=(10, 5))
plt.plot(frames)
plt.title('Frame Blocking')
plt.xlabel('Time')
plt.ylabel('Amplitude')
st.pyplot()

plt.figure(figsize=(10, 5))
plt.plot(fft_frames)
plt.title('FFT Frame Blocking')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
st.pyplot()

st.header("Feature Extraction")
# Convert an amplitude spectrogram to Decibels-scaled spectrogram
st.subheader("Decibels-scaled Spectrogram")
DB = librosa.amplitude_to_db(fft_frames, ref=np.max)
librosa.display.specshow(DB, sr=sample_rate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Decibels-scaled Spectrogram')
plt.xlabel('Time')
plt.ylabel('Hz')
st.pyplot()

# show in dataframe
df = pd.DataFrame(DB)
df = df.iloc[0:4, 0:1024]
st.table(df)

# Convert an amplitude spectrogram to chroma feature
st.subheader("Chroma Feature")
data, sampling_rate = librosa.load('major_minor_audio_dataset_normalized/Major/Major_0.wav')
chromagram = librosa.feature.chroma_stft(y=data, sr=sampling_rate, hop_length=hop_length, n_fft=frame_length)
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.title('Chromagram')
plt.xlabel('Time')
plt.ylabel('Chroma')
plt.colorbar()
st.pyplot()


flatten_chromagram = chromagram.flatten()
chroma_mean = np.mean(flatten_chromagram)
chroma_variance = np.var(flatten_chromagram)
chroma_std = np.std(flatten_chromagram)
db_mean = np.mean(DB)
db_variance = np.var(DB)
db_std = np.std(DB)

# make a dataframe one row and three column (chroma_mean, chroma_variance, chroma_std, db_mean, db_variance, db_std)
df_sample_feature_extraction = pd.DataFrame([['Major_0', chroma_mean, chroma_variance, chroma_std, db_mean, db_variance, db_std, 'Major']], columns=['file', 'chroma_mean', 'chroma_variance', 'chroma_std', 'db_mean', 'db_variance', 'db_std', 'label'])
st.write("Data dibawah ini merupakan contoh hasil ekstraksi fitur pada sample file Major_0.")
st.table(df_sample_feature_extraction)

# # feature extraction pada seluruh data

# df_feature_extraction_major = pd.DataFrame(columns=['file', 'chroma_mean', 'chroma_variance', 'chroma_std', 'db_mean', 'db_variance', 'db_std', 'label'])
# name_list = []
# chroma_mean_list = []
# chroma_variance_list = []
# chroma_std_list = []
# db_mean_list = []
# db_variance_list = []
# db_std_list = []
# label_list = []
# for i in range(len(df_major)):
#     name = 'Major_' + str(i)
    
#     sample_audio, sample_rate = sf.read('major_minor_audio_dataset_normalized/Major/'+name+'.wav')
#     sample_audio_array = np.array(sample_audio)

#     frame_length = 1024
#     hop_length = 512

#     num_frames = 1 + int((len(sample_audio_array) - frame_length) / hop_length)

#     frames = []
#     fft_frames = []
#     for i in range(num_frames):
#         frame = sample_audio_array[i * hop_length : i * hop_length + frame_length]
#         fft_frame = np.fft.fft(frame)
#         fft_frame = np.real(np.abs(fft_frame))
#         fft_frames.append(fft_frame)
#         frames.append(frame)
        
#     fft_frames = np.array(fft_frames)
    
#     DB = librosa.amplitude_to_db(fft_frames, ref=np.max)
    
#     data, sampling_rate = librosa.load('major_minor_audio_dataset_normalized/Major/'+name+'.wav')
#     chromagram = librosa.feature.chroma_stft(y=data, sr=sampling_rate, hop_length=hop_length, n_fft=frame_length)
#     flatten_chromagram = chromagram.flatten()
#     chroma_mean = np.mean(flatten_chromagram)
#     chroma_variance = np.var(flatten_chromagram)
#     chroma_std = np.std(flatten_chromagram)
#     db_mean = np.mean(DB)
#     db_variance = np.var(DB)
#     db_std = np.std(DB)
    
#     df_feature_extraction_major = df_feature_extraction_major.append({'file': name, 'chroma_mean': chroma_mean, 'chroma_variance': chroma_variance, 'chroma_std': chroma_std, 'db_mean': db_mean, 'db_variance': db_variance, 'db_std': db_std, 'label': 'Major'}, ignore_index=True)
    

# st.write("Data dibawah ini merupakan contoh hasil ekstraksi fitur pada seluruh data file Major.")
# st.write(df_feature_extraction_major)


# df_feature_extraction_minor = pd.DataFrame(columns=['file', 'chroma_mean', 'chroma_variance', 'chroma_std', 'db_mean', 'db_variance', 'db_std', 'label'])
# name_list = []
# chroma_mean_list = []
# chroma_variance_list = []
# chroma_std_list = []
# db_mean_list = []
# db_variance_list = []
# db_std_list = []
# label_list = []
# for i in range(len(df_minor)+1):
#     if i != 251:
#         name = 'Minor_' + str(i)
#         sample_audio, sample_rate = sf.read('major_minor_audio_dataset_normalized/Minor/'+name+'.wav')
#         sample_audio_array = np.array(sample_audio)

#         frame_length = 1024
#         hop_length = 512

#         num_frames = 1 + int((len(sample_audio_array) - frame_length) / hop_length)

#         frames = []
#         fft_frames = []
#         for i in range(num_frames):
#             frame = sample_audio_array[i * hop_length : i * hop_length + frame_length]
#             fft_frame = np.fft.fft(frame)
#             fft_frame = np.real(np.abs(fft_frame))
#             fft_frames.append(fft_frame)
#             frames.append(frame)
            
#         fft_frames = np.array(fft_frames)
        
#         DB = librosa.amplitude_to_db(fft_frames, ref=np.max)
        
#         data, sampling_rate = librosa.load('major_minor_audio_dataset_normalized/Minor/'+name+'.wav')
#         chromagram = librosa.feature.chroma_stft(y=data, sr=sampling_rate, hop_length=hop_length, n_fft=frame_length)
#         flatten_chromagram = chromagram.flatten()
#         chroma_mean = np.mean(flatten_chromagram)
#         chroma_variance = np.var(flatten_chromagram)
#         chroma_std = np.std(flatten_chromagram)
#         db_mean = np.mean(DB)
#         db_variance = np.var(DB)
#         db_std = np.std(DB)
        
#         df_feature_extraction_minor = df_feature_extraction_minor.append({'file': name, 'chroma_mean': chroma_mean, 'chroma_variance': chroma_variance, 'chroma_std': chroma_std, 'db_mean': db_mean, 'db_variance': db_variance, 'db_std': db_std, 'label': 'Minor'}, ignore_index=True)
    

# st.write("Data dibawah ini merupakan contoh hasil ekstraksi fitur pada seluruh data file Major.")
# st.write(df_feature_extraction_minor)

# # marge dataframe
# df_final_feature_extraction = pd.concat([df_feature_extraction_major, df_feature_extraction_minor], ignore_index=True)
# st.write("Data dibawah ini merupakan contoh hasil ekstraksi fitur pada seluruh data file Major dan Minor.")
# st.write(df_final_feature_extraction)

# save to csv
# df_final_feature_extraction.to_csv('feature_extraction.csv', index=False)


st.header("Klasifikasi")
st.subheader("Dataset")
df = pd.read_csv('feature_extraction.csv')
st.write(df)
st.subheader("Normalisasi Data")

columns = ['chroma_mean', 'chroma_variance', 'chroma_std', 'db_mean', 'db_variance', 'db_std']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[columns])
scaled_data = pd.DataFrame(scaled_data, columns=columns)
df_scaled = pd.concat([df[['file', 'label']], scaled_data], axis=1)
# make columb label in last
df_scaled = df_scaled[['file', 'chroma_mean', 'chroma_variance', 'chroma_std', 'db_mean', 'db_variance', 'db_std', 'label']]
st.write("Data dibawah ini merupakan hasil normalisasi data.")
st.write(df_scaled)

# Split data
X = df_scaled.drop(['file', 'label'], axis=1)
y = df_scaled['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Data Training")
st.write(X_train)
st.write(y_train)
st.subheader("Data Testing")
st.write(X_test)
st.write(y_test)



# Fungsi untuk menghitung jarak euclidean antara dua vektor
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Fungsi untuk melakukan klasifikasi menggunakan algoritma K-Nearest Neighbors (KNN)
def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            dist = euclidean_distance(X_test[i], X_train[j])
            distances.append((dist, y_train[j]))
        distances = sorted(distances)[:k]
        labels = [d[1] for d in distances]
        y_pred.append(max(set(labels), key=labels.count))
    return y_pred

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

y_pred = knn_predict(X_train, y_train, X_test, 3)

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix")
st.write(cm)

acc = accuracy_score(y_test, y_pred)
st.write("Akurasi : ", acc)