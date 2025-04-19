import dill

import torch
import torch.nn as nn
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ThroughputEstimatorBase(nn.Module):

    def __init__(self):
        super().__init__()

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self = dill.load(f)

    def to_device(self, device = 0):
        self.device = device
        try:
            self.to(device)
            print("Models moved to device successfully.")
        except NameError as e:
            print(f"Error: One or more models are not defined - {e}")

        for name, param in self.named_parameters():
            print(f"Parameter '{name}' is on device: {param.device}")

    def show_summary(self):
        return print(summary(self))

    def test_with_preload_data(self, preloaded_test_data):
        y_noise_level, y_target_throughput_mbps, y_max_throughput_mbps, p_noise_level, p_target_throughput_mbps, p_max_throughput_mbps = [], [], [], [], [], []  # Lists to store ground truth and predictions
        ws = []
        xs = []
        with torch.no_grad():  # Disable gradient computation
            for i, (batch_mac, batch_fapi, batch_iq, batch_y_noise_level, batch_y_target_throughput_mbps, batch_y_max_throughput_mbps) in enumerate(preloaded_test_data):
                # Move to device
                batch_mac = batch_mac.to(self.device)
                batch_iq = batch_iq.to(self.device)
                batch_fapi = batch_fapi.to(self.device)
                # batch_y_noise_level = batch_y_noise_level.to(self.device)
                # batch_y_target_throughput_mbps = batch_y_target_throughput_mbps.to(self.device)
                # batch_y_max_throughput_mbps = batch_y_max_throughput_mbps.to(self.device)
                print("---")
                print(f'batch_fapi: {batch_fapi}')

                x = torch.mean(batch_mac, dim=1)
                x = x.cpu().numpy().squeeze()
                xs.append(x)

                outputs, w = self(batch_mac, batch_fapi, batch_iq)
                print(f'w: {w}')

                predicted = outputs.cpu().numpy().squeeze()  # Handle shape
                targets = batch_y_max_throughput_mbps.cpu().numpy().squeeze()
                w = w.cpu().numpy().squeeze()

                # Collect data
                y_noise_level.append(batch_y_noise_level)
                y_target_throughput_mbps.append(batch_y_target_throughput_mbps)
                y_max_throughput_mbps.append(batch_y_max_throughput_mbps)
                ws.append(w)

                p_noise_level.append(predicted)
                p_target_throughput_mbps.append(predicted)
                p_max_throughput_mbps.append(predicted)

        # Convert to arrays
        y_noise_level = np.array(y_noise_level)
        y_target_throughput_mbps = np.array(y_target_throughput_mbps)
        y_max_throughput_mbps = np.array(y_max_throughput_mbps)
        ws = np.array(ws)

        p_noise_level = np.array(p_noise_level)
        p_target_throughput_mbps = np.array(p_target_throughput_mbps)
        p_max_throughput_mbps = np.array(p_max_throughput_mbps)

        # Verify lengths
        assert len(y_noise_level) == len(p_noise_level), f"Length mismatch : y_noise_level={len(y_noise_level)}, p_noise_level={len(p_noise_level)}"
        print(f"Test set size: {len(y_noise_level)} samples")

        # Compute metrics
        r2_noise_level = r2_score(y_noise_level, p_noise_level)
        mae_noise_level = mean_absolute_error(y_noise_level, p_noise_level)
        mse_noise_level = mean_squared_error(y_noise_level, p_noise_level)
        
        r2_target_throughput_mbps = r2_score(y_target_throughput_mbps, p_target_throughput_mbps)
        mae_target_throughput_mbps = mean_absolute_error(y_target_throughput_mbps, p_target_throughput_mbps)
        mse_target_throughput_mbps = mean_squared_error(y_target_throughput_mbps, p_target_throughput_mbps)

        r2_max_throughput_mbps = r2_score(y_max_throughput_mbps, p_max_throughput_mbps)
        mae_max_throughput_mbps = mean_absolute_error(y_max_throughput_mbps, p_max_throughput_mbps)
        mse_max_throughput_mbps = mean_squared_error(y_max_throughput_mbps, p_max_throughput_mbps)

        print("Metrics noise level:")
        print(f"  R² Score: {r2_noise_level:.4f}")
        print(f"  MAE: {mae_noise_level:.4f}")
        print(f"  MSE: {mse_noise_level:.4f}\n")

        print(f"Metrics target throughput:")
        print(f"  R² Score: {r2_target_throughput_mbps:.4f}")
        print(f"  MAE: {mae_target_throughput_mbps:.4f}")
        print(f"  MSE: {mse_target_throughput_mbps:.4f}\n")

        print(f"Metrics max throughput:")
        print(f"  R² Score: {r2_max_throughput_mbps:.4f}")
        print(f"  MAE: {mae_max_throughput_mbps:.4f}")
        print(f"  MSE: {mse_max_throughput_mbps:.4f}\n")
        
        return y_noise_level, p_noise_level, y_target_throughput_mbps, p_target_throughput_mbps, y_max_throughput_mbps, p_max_throughput_mbps, ws, xs

    def show_test(self, preloaded_test_data, step=1, start_idx=None, stop_idx=None):
        y_noise_level, p_noise_level, y_target_throughput_mbps, p_target_throughput_mbps, y_max_throughput_mbps, p_max_throughput_mbps, ws, xs = self.test_with_preload_data(preloaded_test_data)

        if (start_idx is not None and stop_idx is not None):
            y_noise_level = y_noise_level[start_idx:stop_idx]
            p_noise_level = p_noise_level[start_idx:stop_idx]
            y_target_throughput_mbps = y_target_throughput_mbps[start_idx:stop_idx]
            p_target_throughput_mbps = p_target_throughput_mbps[start_idx:stop_idx]
            y_max_throughput_mbps = y_max_throughput_mbps[start_idx:stop_idx]
            p_max_throughput_mbps = p_max_throughput_mbps[start_idx:stop_idx]
            ws = ws[start_idx:stop_idx]
            xs = xs[start_idx:stop_idx]

        y_sub_noise_level = y_noise_level[::step]
        p_sub_noise_level = p_noise_level[::step]

        y_sub_target_throughput_mbps = y_target_throughput_mbps[::step]
        p_sub_target_throughput_mbps = p_target_throughput_mbps[::step]

        y_sub_max_throughput_mbps = y_max_throughput_mbps[::step]
        p_sub_max_throughput_mbps = p_max_throughput_mbps[::step]
        
        ws_sub = ws[::step]
        xs_sub = xs[::step]
        
        time_steps = np.arange(len(y_noise_level))

        # Plotting
        fig, axs = plt.subplots(5, 1, figsize=(12, 15))
        
        # Noise Level Plot
        axs[0].plot(time_steps, y_sub_noise_level, label='Noise Level')
        axs[0].set_title('Noise Level')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Noise Level')
        axs[0].grid(True)
        axs[0].legend()
        
        # Target Throughput Plot
        axs[1].plot(time_steps, y_sub_target_throughput_mbps, label='Target Throughput')
        axs[1].set_title('Target Throughput')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Throughput (Mbps)')
        axs[1].grid(True)
        axs[1].legend()
        
        # Max Throughput Plot
        axs[2].plot(time_steps, y_sub_max_throughput_mbps, label='Measured Max Throughput')
        axs[2].plot(time_steps, p_sub_max_throughput_mbps, label='Estimated Max Throughput', linestyle='--')
        axs[2].set_title('Max Throughput')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Throughput (Mbps)')
        axs[2].grid(True)
        axs[2].legend()
        
        # Weight Plot
        axs[3].plot(time_steps, ws_sub, label='Weight')
        axs[3].set_title('Weight')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Weight')
        axs[3].grid(True)
        axs[3].legend()

        # XS Plot
        axs[4].plot(time_steps, xs_sub, label='XS')
        axs[4].set_title('XS')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('XS')
        axs[4].grid(True)
        axs[4].legend()
        
        plt.tight_layout()
        plt.show()

class ThroughputEstimatorMacIq(ThroughputEstimatorBase):

    def __init__(self, lstm_input_size, output_size, hidden_size, num_layers, drop_out=0.1):
        super().__init__()

        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.drop_out = drop_out

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.drop_out if self.num_layers > 1 else 0.0,
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 819 * 3, self.hidden_size),  # Verify dimensions
            nn.ReLU(),
            nn.Dropout(self.drop_out)
        )

        # Unified FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )

    def forward(self, lstm_input, fapi, cnn_input):
        # LSTM processing
        h0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = lstm_out[:, -1, :]

        # CNN processing
        cnn_out = self.cnn(cnn_input)

        # Weighted combination
        w = (fapi[:, 1] * fapi[:, 3]).unsqueeze(-1)
        combined = w * lstm_out + (1 - w) * cnn_out
        out = self.fc(combined)

        return out.squeeze(-1) if self.output_size == 1 else out, w


class ThroughputEstimatorMac(ThroughputEstimatorBase):

    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_out=0.1):
        super().__init__()
        
        self.lstm_input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.drop_out if self.num_layers > 1 else 0.0,
        )

        # Unified FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
        
    def forward(self, lstm_input, fapi, cnn_input):
        w = (fapi[:, 1] * fapi[:, 3]).unsqueeze(-1)

        # LSTM processing
        h0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = lstm_out[:, -1, :]

        out = self.fc(lstm_out)
        return out.squeeze(-1) if self.output_size == 1 else out, w


class ThroughputEstimatorIq(ThroughputEstimatorBase):

    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_out=0.1):
        super().__init__()
        
        self.cnn_input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 819 * 3, self.hidden_size),  # Verify dimensions
            nn.ReLU(),
            nn.Dropout(self.drop_out)
        )

        # Unified FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
        
    def forward(self, lstm_input, fapi, cnn_input):
        w = (fapi[:, 1] * fapi[:, 3]).unsqueeze(-1)

        # CNN processing
        cnn_out = self.cnn(cnn_input)
        cnn_out = cnn_out.squeeze(-1)

        out = self.fc(cnn_out)
        return out.squeeze(-1) if self.output_size == 1 else out, w
