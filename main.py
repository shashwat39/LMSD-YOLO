import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_iq_data(rx_file, tx_file):
    """Load RX and TX IQ data from CSV files."""
    rx_data = pd.read_csv(rx_file)
    tx_data = pd.read_csv(tx_file)

    rx_signal = rx_data['I'] + 1j * rx_data['Q']  # Combine I and Q to create a complex signal
    tx_signal = tx_data['I'] + 1j * tx_data['Q']  # Combine I and Q for the transmitter signal

    return np.array(rx_signal), np.array(tx_signal)

def extract_features_with_labels(signal_ref, signal_meas, window_size, shift, dt, threshold_ratio=0.5):
    """Extract features with automated labels using cross-correlation peaks."""
    n = len(signal_ref)
    num_windows = int((n - window_size) / shift + 1)
    features = []

    for ci in range(num_windows):
        start_index = ci * shift
        end_index = start_index + window_size

        if end_index > n:
            break

        window_ref = signal_ref[start_index:end_index]
        window_meas = signal_meas[start_index:end_index]

        corr = correlate(window_ref, window_meas, mode='full')
        peak_amplitude = np.max(np.abs(corr))
        peak_index = np.argmax(np.abs(corr)) - (len(corr) // 2)
        time_delay = peak_index * dt

        threshold = threshold_ratio * peak_amplitude
        label = 1 if peak_amplitude > threshold else 0

        features.append([peak_amplitude, time_delay, label])

    return np.array(features)

def visualize_cross_correlation(signal_ref, signal_meas, window_size, dt, threshold_ratio=0.5):
    """Visualize cross-correlation and highlight peaks."""
    window_ref = signal_ref[:window_size]
    window_meas = signal_meas[:window_size]

    corr = correlate(window_ref, window_meas, mode='full')
    lags = np.arange(-len(window_ref) + 1, len(window_ref))

    peak_amplitude = np.max(np.abs(corr))
    peak_index = np.argmax(np.abs(corr))
    time_delay = (peak_index - len(window_ref) + 1) * dt

    plt.figure(figsize=(10, 6))
    plt.plot(lags * dt, np.abs(corr), label='Cross-Correlation')
    plt.scatter(lags[peak_index] * dt, peak_amplitude, color='red', label='Detected Peak', zorder=5)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(x=time_delay, color='red', linestyle='--', label=f"Time Delay = {time_delay:.3e} seconds")
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation Amplitude')
    plt.title('Cross-Correlation with Detected Peak')
    plt.legend()
    plt.show()

def prepare_dataset(signal_ref, signal_meas, window_size=100, shift=50, dt=1e-3):
    """Generate features and labels from signals."""
    features = extract_features_with_labels(signal_ref, signal_meas, window_size, shift, dt)
    X = features[:, :2]  # Peak amplitude and time delay as features
    y = features[:, 2]   # Labels (1 = object detected, 0 = no object detected)
    return X, y

def train_model(X, y):
    """Train a Random Forest model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return clf

def visualize_results(X, y, clf):
    """Visualize decision boundaries and results."""
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='True Labels')
    plt.colorbar(label='Label (0=No Object, 1=Object)')
    plt.xlabel('Peak Amplitude')
    plt.ylabel('Time Delay')
    plt.title('Feature Space with Labels')
    plt.legend()
    plt.show()

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
    plt.title('Decision Boundary')
    plt.xlabel('Peak Amplitude')
    plt.ylabel('Time Delay')
    plt.show()

def main():
    rx_file = 'rx_IQ_data.csv'
    tx_file = 'tx_IQ_data.csv'

    signal_ref, signal_meas = load_iq_data(rx_file, tx_file)

    plt.figure(figsize=(10, 4))
    plt.plot(np.real(signal_ref), label='RX Signal (Real)')
    plt.plot(np.real(signal_meas), label='TX Signal (Real)', alpha=0.8)
    plt.legend()
    plt.title('IQ Signals (Real Part)')
    plt.show()

    visualize_cross_correlation(signal_ref, signal_meas, window_size=200, dt=1e-3)

    X, y = prepare_dataset(signal_ref, signal_meas, window_size=200, shift=100, dt=1e-9)
    clf = train_model(X, y)
    visualize_results(X, y, clf)

if __name__ == "__main__":
    main()
