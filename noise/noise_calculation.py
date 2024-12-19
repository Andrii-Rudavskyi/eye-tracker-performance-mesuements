import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



def extract_noise(signal, axt, dt, pl=1, data_label = '', results_path = './'):
    # Step 1: Create the filter manually
    alpha = 1 - np.exp(-np.diff(axt) / dt)  # Compute alpha for each interval
    alpha = np.insert(alpha, 0, alpha[0])  # Match the length with the signal

    # Initialize filtered signal
    filtered_signal = np.zeros_like(signal)
    filtered_signal[0] = signal[0]  # First value remains the same

    # Step 2: Apply the filter using a recursive approach
    for i in range(1, len(signal)):
        filtered_signal[i] = (
            alpha[i] * signal[i] + (1 - alpha[i]) * filtered_signal[i - 1]
        )

    # Step 3: Calculate the noise
    noise = signal - filtered_signal

    # Step 4: Calculate average noise on a trace
    average_noise = np.std(noise)

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    if pl:
        # Plot the results
        plt.figure(figsize=(10, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(axt, signal)
        plt.title('Original Signal, ' + data_label)
        plt.ylim([np.min(signal), np.max(signal)])

        plt.subplot(3, 1, 2)
        plt.plot(axt, filtered_signal)
        plt.title('Filtered Signal, ' + data_label)
        plt.ylim([np.min(signal), np.max(signal)])

        plt.subplot(3, 1, 3)
        plt.plot(axt, noise)
        plt.title(f"Noise -- std = {np.std(noise):.4f}")

        plt.tight_layout()
        plt.savefig(results_path + data_label)


    return filtered_signal, noise, average_noise


data_source = './traces/'
output_dir = './results/'

for dirs in os.listdir(data_source):
    headpath = data_source + dirs + '/' + dirs + '_headPose.csv'
    eyesdatapath = data_source + dirs + '/' + dirs + '_eyeCoordinates.csv'
    rawpath = data_source + dirs + '/' + dirs + '_raw.csv'
    predictedpath = data_source + dirs + '/' + dirs + '_predicted.csv'


    if (not os.path.exists(headpath)):
        print(headpath)
        print("File does not exist does not exist")
    else:
        hedapose_data = pd.read_csv(headpath)

    if (not os.path.exists(eyesdatapath)):
        print(eyesdatapath)
        print("File does not exist does not exist")
    else:
        eyes_data = pd.read_csv(eyesdatapath)

    if (not os.path.exists(rawpath)):
        print(rawpath)
        print("File does not exist does not exist")
    else:
        raw_data = pd.read_csv(rawpath)

    if (not os.path.exists(predictedpath)):
        print(predictedpath)
        print("File does not exist does not exist")
    else:
        predicted_data = pd.read_csv(predictedpath)

    traces = []
    labels = []
    raw_traces = []
    raw_labels = []
    predicted_traces = []
    predicted_labels = []
    timestamps = 1000 * hedapose_data['CaptureTime'] #converst to from seconds to milliseconds
    raw_timestamps = 1000 * raw_data[' timeCaptured']
    predicted_timestamps = 1000 * predicted_data['timeLogged']
    #trace_center_Y2D = 0.5 * ( eyes_data[' Camera1Left2D.y'] + eyes_data[' Camera1Right2D.y'] )

    #x2D
    trace = 0.5 * ( eyes_data[' Camera1Left2D.x'] + eyes_data[' Camera1Right2D.x'] )
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('X2D_camera1')

    trace = 0.5 * ( eyes_data[' Camera2Left2D.x'] + eyes_data[' Camera2Right2D.x'] )
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('X2D_camera2')

    #y2x
    trace = 0.5 * ( eyes_data[' Camera1Left2D.y'] + eyes_data[' Camera1Right2D.y'] )
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('Y2D_camera1')

    trace = 0.5 * ( eyes_data[' Camera2Left2D.y'] + eyes_data[' Camera2Right2D.y'] )
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('Y2D_camera2')


    #Headpose Z
    trace = hedapose_data[' Camera1HeadPosition.z']
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('headposeZ_camera1')

    trace = hedapose_data[' Camera2HeadPosition.z']
    if (len(trace) != 0):
        traces.append(trace)
        labels.append('headposeZ_camera2')

    #Raw
    trace = 0.5 * (raw_data[' leftEye.x'] + raw_data[' rightEye.x']) * 10 #Convert to mm
    if (len(trace) != 0):
        raw_traces.append(trace)
        raw_labels.append('X3D_raw')

    trace = 0.5 * (raw_data[' leftEye.y'] + raw_data[' rightEye.y']) * 10
    if (len(trace) != 0):
        raw_traces.append(trace)
        raw_labels.append('Y3D_raw')

    trace = 0.5 * (raw_data[' leftEye.z'] + raw_data[' rightEye.z']) * 10
    if (len(trace) != 0):
        raw_traces.append(trace)
        raw_labels.append('Z3D_raw')

    # Predicted
    trace = 0.5 * (predicted_data[' leftEye.x'] + predicted_data[' rightEye.x']) * 10
    if (len(trace) != 0):
        predicted_traces.append(trace)
        predicted_labels.append('X3D_predicted')

    trace = 0.5 * (predicted_data[' leftEye.y'] + predicted_data[' rightEye.y']) * 10
    if (len(trace) != 0):
        predicted_traces.append(trace)
        predicted_labels.append('Y3D_predicted')

    trace = 0.5 * (predicted_data[' leftEye.z'] + predicted_data[' rightEye.z']) * 10
    if (len(trace) != 0):
        predicted_traces.append(trace)
        predicted_labels.append('Z3D_predicted')


    #loop through all signals
    for i in range(0, len(labels)):
        signal = traces[i]
        filtered_signal, noise, noise_on_trace = extract_noise(signal=signal, axt=timestamps, dt=100, pl=1, data_label=labels[i], results_path = output_dir + dirs + '/')
        print(labels[i], noise_on_trace)

    for i in range(0, len(raw_labels)):
        signal = raw_traces[i]
        filtered_signal, noise, noise_on_trace = extract_noise(signal=signal, axt=raw_timestamps, dt=100, pl=1, data_label=raw_labels[i], results_path = output_dir + dirs + '/')
        print(raw_labels[i], noise_on_trace)

    for i in range(0, len(predicted_labels)):
        signal = predicted_traces[i]
        filtered_signal, noise, noise_on_trace = extract_noise(signal=signal, axt=predicted_timestamps, dt=100, pl=1, data_label=predicted_labels[i], results_path = output_dir + dirs + '/')
        print(predicted_labels[i], noise_on_trace)

