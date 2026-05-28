
import scipy.io
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.table import Table

mat_data = scipy.io.loadmat("TMP117 - room plus freezer.mat")

def plot(mat_data, struct_key='TMP117', time_field='Time', raw_field='raw', degc_field='degC'):
    import numpy as np
    import matplotlib.pyplot as plt
    # Try to extract the structured array
    if struct_key not in mat_data:
        print(f"Key '{struct_key}' not found in .mat file.")
        return
    data = mat_data[struct_key]
    if not hasattr(data, 'dtype') or not data.dtype.names:
        print(f"Key '{struct_key}' is not a structured array.")
        return
    # Extract fields (handle nested object arrays)
    def extract_first_array(field):
        arr = data[field]
        # If arr is object dtype and shape (1, 1), get the first element
        if hasattr(arr, 'dtype') and arr.dtype == object:
            arr = arr[0, 0]
        return arr.squeeze()

    try:
        time = extract_first_array(time_field)
        raw = extract_first_array(raw_field)
        degc = extract_first_array(degc_field)
    except Exception as e:
        print(f"Error extracting fields: {e}")
        return
    # Plot raw vs time
    plt.figure()
    plt.plot(time, raw)
    plt.xlabel('Time')
    plt.ylabel('Raw')
    plt.title('Raw vs Time')
    plt.grid(True)
    plt.show()
    # Plot degC vs time
    plt.figure()
    plt.plot(time, degc)
    plt.xlabel('Time')
    plt.ylabel('degC')
    plt.title('degC vs Time')
    plt.grid(True)
    plt.show()

def freq_analysis():
    import numpy as np
    import matplotlib.pyplot as plt
    struct_key = 'TMP117'
    raw_field = 'raw'
    time_field = 'Time'
    # Extract structured array
    if struct_key not in mat_data:
        print(f"Key '{struct_key}' not found in .mat file.")
        return
    data = mat_data[struct_key]
    if not hasattr(data, 'dtype') or not data.dtype.names:
        print(f"Key '{struct_key}' is not a structured array.")
        return
    def extract_first_array(field):
        arr = data[field]
        if hasattr(arr, 'dtype') and arr.dtype == object:
            arr = arr[0, 0]
        return arr.squeeze()

    try:
        raw = extract_first_array(raw_field)
        time = extract_first_array(time_field)
    except Exception as e:
        print(f"Error extracting fields: {e}")
        return

    # Calculate sampling frequency
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    n = len(raw)
    # Remove mean for FFT
    raw_centered = raw - np.mean(raw)
    # Compute FFT
    fft_vals = np.fft.fft(raw_centered)
    fft_freqs = np.fft.fftfreq(n, d=dt)
    # Only plot positive frequencies
    idx = np.where(fft_freqs >= 0)
    plt.figure()
    plt.plot(fft_freqs[idx], np.abs(fft_vals[idx]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Raw Data')
    plt.grid(True)
    plt.show()

def main():
    plot(mat_data)
    freq_analysis()



if __name__ == "__main__":
    main()
