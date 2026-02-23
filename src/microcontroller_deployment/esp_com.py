import serial
import time
import csv
import struct
import numpy as np
from tqdm import tqdm

def establish_connection(port, baud_rate):
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to {port} at {baud_rate} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def send_data(ser, data):
    if ser:
        ser.write(data)
        # print(f"Sent: {data}")

def receive_data(ser):
    if ser:
        data = ser.readline().decode().strip()
        # print(f"Received: {data}")
        return data

if __name__ == "__main__":
    port = "COM9"  # Replace with your port
    baud_rate = 115200
    ser = establish_connection(port, baud_rate)
    y_preds = []
    y_trues = []
    accuracy = 0
    
    if ser:
        #Read the test_df.csv file
        with open('test_df.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            rows = list(reader)
            for row in tqdm(rows, desc="Processing rows"):
                irms = float(row[0])
                pf = float(row[1]) 
                p = float(row[2]) 
                q = float(row[3]) 
                s = float(row[4]) 
                label = int(row[5]) 
                # Convert each value to 32-bit float and pack into bytes
                irms_bytes = struct.pack('f', irms)
                pf_bytes   = struct.pack('f', pf)
                p_bytes    = struct.pack('f', p)
                q_bytes    = struct.pack('f', q)
                s_bytes    = struct.pack('f', s)

                # Combine all bytes into a single message
                # Order defined by ICTA_winter.ino:
                # receivedData[0] -> irms
                # receivedData[1] -> realPower (P)
                # receivedData[2] -> powerFactor (PF)
                # receivedData[3] -> apparentPower (S)
                # receivedData[4] -> reactivePower (Q)
                # Note: original code sent Irms, PF, P, Q, S which was mapped incorrectly.
                # Corrected implementation sends: Irms, P, PF, S, Q
                message = irms_bytes + p_bytes + pf_bytes + s_bytes + q_bytes
                send_data(ser, message)
                time.sleep(0.1)
                data = receive_data(ser)
                pred = int(data)
                y_preds.append(pred)
                y_trues.append(label)

        # Calculate accuracy
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        accuracy = np.mean(y_preds == y_trues) * 100
        print(f"Accuracy with ESP32: {accuracy:.2f}%")
        
                    
            