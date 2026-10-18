import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class EEGProcessor:
    """
    A foundational pipeline for EEG data processing and machine learning evaluation.
    
    Provides an object-oriented interface to seamlessly load, visualize, preprocess 
    raw EEG signals, and evaluate machine learning models.
    """


    # __init__ is a Dunder Methods , which is Double Underscore
    def __init__(self, subject=1, runs=4):
        """
        Initializes the processor and binds it to specific subject data.
        
        Args:
            subject (int): The subject identifier (e.g., 1).
            runs (int): The experimental run number (e.g., 4).

        _____________________________________________________________________________________________________________________________________

        Methods:
            load_data(): Downloads and loads the corresponding EDF format medical data into memory.
            filter_data(low_freq=1.0, high_freq=40.0): Applies a band-pass filter to the raw EEG data to remove noise.
            extract_features(window_size=1.0): Slices the continuous waveform into epochs and calculates variance as features.
            train_and_evaluate(model=None): Evaluates a given Scikit-Learn model using 5-fold cross-validation.

            
        """
        self.subject = subject
        self.runs = runs
        self.raw = None
        self.features = None
        self.labels = None

    def load_data(self):
        """Downloads and loads the corresponding EDF format medical data into memory."""
        data_path = mne.datasets.eegbci.load_data(subject=self.subject, runs=self.runs)
        self.raw = mne.io.read_raw_edf(data_path[0], preload=True, verbose=False)
        print("✅ Step 1: Raw EEG data loaded successfully.")

    def filter_data(self, low_freq=1.0, high_freq=40.0):
        """Applies a band-pass filter to the raw EEG data to remove noise."""
        if self.raw is not None:
            self.raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
            print(f"✅ Step 2: Filtering completed ({low_freq}Hz - {high_freq}Hz).")
        else:
            print("Error: No raw data to filter. Please call load_data() first.")

    def extract_features(self, window_size=1.0):
        """
        Slices the continuous waveform into epochs and calculates variance as features.
        (This is the slot where Tommy will insert his proprietary feature extraction logic).
        """
        if self.raw is None:
            print("Error: No data available for feature extraction.")
            return
            
        # 把波形按 1 秒切割
        epochs = mne.make_fixed_length_epochs(self.raw, duration=window_size, preload=True, verbose=False)
        data_matrix = epochs.get_data(copy=False)
        
        # 提取特征：计算每个通道在 1 秒内的方差
        self.features = np.var(data_matrix, axis=2) 
        
        # 模拟业务标签：随机生成 0 (健康) 和 1 (发病)
        self.labels = np.random.randint(0, 2, size=len(self.features))
        
        print(f"✅ Step 3: Feature extraction completed. Generated {len(self.features)} samples with {self.features.shape[1]} features each.")

    def train_and_evaluate(self, model=None):
        """
        Evaluates a given Scikit-Learn model using 5-fold cross-validation.
        
        Args:
            model: A Scikit-Learn estimator instance. Defaults to RandomForestClassifier.
        """
        if self.features is None or self.labels is None:
            print("Error: Please run extract_features() first.")
            return
            
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("⏳ No model specified. Defaulting to RandomForestClassifier.")
        else:
            print(f"⏳ Using specified model: {model.__class__.__name__}")
        
        # 5折交叉验证
        scores = cross_val_score(model, self.features, self.labels, cv=5)
        
        print("=======================================")
        print(f"🎯 Evaluation Results for {model.__class__.__name__}:")
        print(f"Scores across 5 folds: {np.round(scores, 3)}")
        print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
        print("=======================================")