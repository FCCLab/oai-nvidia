import os
import glob
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average(arr, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(arr, weights, mode='valid')

def get_throughput(arr):
    W = 10
    mv_avg = moving_average(arr, W)
    largest_numbers = heapq.nlargest(W, mv_avg)
    return sum(largest_numbers) / W

def load_dataset_v2(DATASET_PATH):
    dataset = pd.DataFrame()

    wgn_file_paths = glob.glob(os.path.join(DATASET_PATH, 'df_*_wgn.parquet'))
    print(wgn_file_paths)

    test_indces = [(wgn_file_path.split('/')[-1].split('_')[1]) for wgn_file_path in wgn_file_paths]
    test_indces.sort()
    print(test_indces)

    for test_index in test_indces:
        print(f'Test index: {test_index}')

        wgn_file_paths = glob.glob(os.path.join(DATASET_PATH, f'df_{test_index}_*_wgn.parquet'))
        if len(wgn_file_paths) != 1:
            print(f'Found {len(wgn_file_paths)} files for test index {test_index}: {wgn_file_paths}')
            continue
        wgn_file_path = wgn_file_paths[0]
        print(f'WGN file path: {wgn_file_path}')

        mac_file_paths = glob.glob(os.path.join(DATASET_PATH, f'df_{test_index}_*_mac.parquet'))
        if len(mac_file_paths) != 1:
            print(f'Found {len(mac_file_paths)} files for test index {test_index}: {mac_file_paths}')
            continue
        mac_file_path = mac_file_paths[0]
        print(f'MAC file path: {mac_file_path}')

        fapi_file_paths = glob.glob(os.path.join(DATASET_PATH, f'df_{test_index}_*_fapi.parquet'))
        if len(fapi_file_paths) != 1:
            print(f'Found {len(fapi_file_paths)} files for test index {test_index}: {fapi_file_paths}')
            continue
        fapi_file_path = fapi_file_paths[0]
        print(f'FAPI file path: {fapi_file_path}')

        fh_file_paths = glob.glob(os.path.join(DATASET_PATH, f'df_{test_index}_*_fh.parquet'))
        if len(fh_file_paths) != 1:
            print(f'Found {len(fh_file_paths)} files for test index {test_index}: {fh_file_paths}')
            continue
        fh_file_path = fh_file_paths[0]
        print(f'FH file path: {fh_file_path}')

        wgn_df = pd.read_parquet(wgn_file_path)
        # print(wgn_df)
        throughput_mbps = wgn_df['ThroughputMbps'].values
        # print(throughput_mbps)
        # plt.plot(throughput_mbps)
        # plt.show()
        max_throughput = get_throughput(throughput_mbps)
        print(f'Max throughput: {max_throughput}')
        
        mac_df = pd.read_parquet(mac_file_path)
        # print(mac_df)
        fapi_df = pd.read_parquet(fapi_file_path)
        # print(fapi_df)
        fh_df = pd.read_parquet(fh_file_path)
        # print(fh_df)

        for fapi_idx, fapi in fapi_df.iterrows():
            TsTaiNs = fapi.TsTaiNs
            print(f'Sample at {TsTaiNs}')

            matching_mac = mac_df[mac_df['TsTaiNs'] <= TsTaiNs]
            # print(matching_mac)
            if len(matching_mac) < 30:
                continue

            # Sort by TsTaiNs in descending order and get the 10 latest records
            matching_mac = matching_mac.sort_values(by='TsTaiNs', ascending=False).head(30)
            # print(matching_mac)
            matching_mac = matching_mac[['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']].to_numpy()
            # print(matching_mac.shape) # (20, 14)
            # print(matching_mac)
            # Assuming matching_mac is the NumPy array with shape (n_samples, 14)
            # Columns: ['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 
            #           'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']

            # Assuming matching_mac is the NumPy array with shape (n_samples, 14)
            # Columns: ['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 
            #           'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']

            # Step 1: Compute HARQ differences (harq_new - harq_old)
            # harq_new is first row, harq_old is each subsequent row
            harq_indices = [10, 11, 12, 13]  # Indices of ul_harq_0 to ul_harq_3
            modified_matching_mac = matching_mac.copy()  # Avoid modifying original array

            # Compute differences: first row (harq_new) - current row (harq_old)
            for idx in harq_indices:
                modified_matching_mac[:, idx] = matching_mac[0, idx] - matching_mac[:, idx]
                # Handle uint32 wrap-around (if old > new, difference might be negative)
                modified_matching_mac[:, idx] = np.where(
                    modified_matching_mac[:, idx] < 0,
                    modified_matching_mac[:, idx] + 2**32,
                    modified_matching_mac[:, idx]
                )
            # First row's HARQ differences set to 0 (no difference from itself)
            modified_matching_mac[0, harq_indices] = 0

            # Step 2: Define fixed min/max values for normalization
            fixed_max_values = np.array([
                -44,  # ss_rsrp (dBm)
                -3,   # ss_rsrq (dB)
                30,   # ss_sinr (dB)
                4,    # ri
                15,   # wb_cqi_1tb
                7,    # cri
                40,   # phr (dB)
                30,   # pusch_snr (dB)
                31,   # ul_mcs1
                1,    # ul_bler
                1000,  # ul_harq_0 (max difference in retransmissions, adjust as needed)
                1000,  # ul_harq_1
                1000,  # ul_harq_2
                1000   # ul_harq_3
            ])

            fixed_min_values = np.array([
                -140,  # ss_rsrp
                -20,   # ss_rsrq
                -20,   # ss_sinr
                1,     # ri
                0,     # wb_cqi_1tb
                0,     # cri
                -23,   # phr
                -20,   # pusch_snr
                0,     # ul_mcs1
                0,     # ul_bler
                0,     # ul_harq_0 (min difference)
                0,     # ul_harq_1
                0,     # ul_harq_2
                0      # ul_harq_3
            ])

            # Step 3: Min-Max normalization with fixed values
            range_vals = fixed_max_values - fixed_min_values
            range_vals[range_vals == 0] = 1  # Prevent division by zero

            normalized_matching_mac = (modified_matching_mac - fixed_min_values) / range_vals
            # print(normalized_matching_mac)

            matching_fh = fh_df[fh_df['TsTaiNs'] == TsTaiNs]
            # print(len(matching_fh))
            if len(matching_fh) != 1:
                continue
            # print(matching_fh)

            fh_samp = np.array(matching_fh['fhData'].iloc[0], dtype=np.float32)
            rx_slot = np.swapaxes(fh_samp.view(np.complex64).reshape(4, 14, 273*12), 2, 0)[:,:,0]
            # print(rx_slot)
            # print(rx_slot.shape)

            # rx_slot[fapi.rbStart: fapi.rbSize * 12, fapi.StartSymbolIndex : 14] = 0
            I = np.real(rx_slot)  # Real part
            Q = np.imag(rx_slot)  # Imaginary part
            IQ_tensor = np.stack([I, Q], axis=0)
            # print(IQ_tensor.shape) # (2, 3276, 14)

            # # fig, axs = plt.subplots(1)
            # # axs.imshow(10*np.log10(np.abs(rx_slot**2)), aspect='auto')
            # # axs.set_ylim([0, 273 * 12])
            # # axs.set_xlim([0, 14])
            # # axs.set_title('Ant ' + str(0))
            # # axs.set(xlabel='Symbol', ylabel='Resource Element')
            # # axs.label_outer()
            # # fig.suptitle('Power in RU Antennas') 
            # # plt.show(fig)
            
            matching_wgn = wgn_df[wgn_df['TsTaiNs'] <= TsTaiNs]
            matching_wgn = matching_wgn.sort_values(by='TsTaiNs', ascending=False).head(1)
            # print(matching_wgn['target_throughput_mbps'].to_numpy()[0])
            distance = matching_wgn['AdjDistanceDb'].to_numpy()[0]
            noise_level_db = matching_wgn['NoiseLevelDb'].to_numpy()[0]
            target_throughput_mbps = matching_wgn['TargetThroughputMbps'].to_numpy()[0]

            new_row = pd.DataFrame({
                'AdjDistanceDb': [distance],
                'NoiseLevelDb': [noise_level_db],
                'TargetThroughputMbps': [target_throughput_mbps],
                'MaxThroughputMbps': [max_throughput],
                'mac': [normalized_matching_mac],
                'fapi': [[fapi.rbStart/273, fapi.rbSize/273, fapi.StartSymbolIndex/14, 14/14]],
                'iq': [IQ_tensor],
            })
            dataset = pd.concat([dataset, new_row], ignore_index=True)
            # break
        # break
    
    return dataset

def load_dataset_v1(DATASET_PATH):
    # Noise level
    # -- target_throughput_mbps

    wgn_file_paths = glob.glob(os.path.join(DATASET_PATH, 'df_wgn_*.parquet'))
    print(wgn_file_paths)

    noise_levels = [float(wgn_file_path.replace('.parquet', '').split('_')[-1]) for wgn_file_path in wgn_file_paths]
    noise_levels.sort()
    print(noise_levels)

    dataset = pd.DataFrame()

    for noise_level_idx, noise_level in enumerate(noise_levels):
        if noise_level_idx % 2 == 1:
            continue
        print(f'Noise level: {noise_level}')

        wgn_files = glob.glob(os.path.join(DATASET_PATH, f'df_wgn_*{noise_level}*.parquet'))
        if len(wgn_files) != 1:
            print(f'Found {len(wgn_files)} files for noise level {noise_level}')
            continue
        wgn_file_path = wgn_files[0]
        wgn_df = pd.read_parquet(wgn_file_path)
        # print(wgn_df)
        throughput_mbps = wgn_df['throughput_mbps'].values
        # print(throughput_mbps)
        # plt.plot(throughput_mbps)
        # plt.show()
        throughput = get_throughput(throughput_mbps)
        
        mac_files = glob.glob(os.path.join(DATASET_PATH, f'df_mac_*{noise_level}*.parquet'))
        if len(mac_files) != 1:
            print(f'Found {len(mac_files)} files for noise level {noise_level}')
            continue
        mac_file_path = mac_files[0]
        mac_df = pd.read_parquet(mac_file_path)
        # print(mac_df)

        fapi_files = glob.glob(os.path.join(DATASET_PATH, f'df_fapi_*{noise_level}*.parquet'))
        if len(fapi_files) != 1:
            print(f'Found {len(fapi_files)} files for noise level {noise_level}')
            continue
        fapi_file_path = fapi_files[0]
        fapi_df = pd.read_parquet(fapi_file_path)
        # print(fapi_df)
        
        fh_files = glob.glob(os.path.join(DATASET_PATH, f'df_fh_*{noise_level}*.parquet'))
        if len(fh_files) != 1:
            print(f'Found {len(fh_files)} files for noise level {noise_level}')
            continue
        fh_file_path = fh_files[0]
        fh_df = pd.read_parquet(fh_file_path)
        # print(fh_df)

        for fapi_idx, fapi in fapi_df.iterrows():
            TsTaiNs = fapi.TsTaiNs
            print(f'Sample at {TsTaiNs}')

            matching_mac = mac_df[mac_df['TsTaiNs'] <= TsTaiNs]
            # print(matching_mac)
            if len(matching_mac) < 30:
                continue

            # Sort by TsTaiNs in descending order and get the 10 latest records
            matching_mac = matching_mac.sort_values(by='TsTaiNs', ascending=False).head(30)
            # print(matching_mac)
            matching_mac = matching_mac[['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']].to_numpy()
            # print(matching_mac.shape) # (20, 14)
            # print(matching_mac)
            # Assuming matching_mac is the NumPy array with shape (n_samples, 14)
            # Columns: ['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 
            #           'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']

            # Assuming matching_mac is the NumPy array with shape (n_samples, 14)
            # Columns: ['ss_rsrp', 'ss_rsrq', 'ss_sinr', 'ri', 'wb_cqi_1tb', 'cri', 'phr', 
            #           'pusch_snr', 'ul_mcs1', 'ul_bler', 'ul_harq_0', 'ul_harq_1', 'ul_harq_2', 'ul_harq_3']

            # Step 1: Compute HARQ differences (harq_new - harq_old)
            # harq_new is first row, harq_old is each subsequent row
            harq_indices = [10, 11, 12, 13]  # Indices of ul_harq_0 to ul_harq_3
            modified_matching_mac = matching_mac.copy()  # Avoid modifying original array

            # Compute differences: first row (harq_new) - current row (harq_old)
            for idx in harq_indices:
                modified_matching_mac[:, idx] = matching_mac[0, idx] - matching_mac[:, idx]
                # Handle uint32 wrap-around (if old > new, difference might be negative)
                modified_matching_mac[:, idx] = np.where(
                    modified_matching_mac[:, idx] < 0,
                    modified_matching_mac[:, idx] + 2**32,
                    modified_matching_mac[:, idx]
                )
            # First row's HARQ differences set to 0 (no difference from itself)
            modified_matching_mac[0, harq_indices] = 0

            # Step 2: Define fixed min/max values for normalization
            fixed_max_values = np.array([
                -44,  # ss_rsrp (dBm)
                -3,   # ss_rsrq (dB)
                30,   # ss_sinr (dB)
                4,    # ri
                15,   # wb_cqi_1tb
                7,    # cri
                40,   # phr (dB)
                30,   # pusch_snr (dB)
                31,   # ul_mcs1
                1,    # ul_bler
                1000,  # ul_harq_0 (max difference in retransmissions, adjust as needed)
                1000,  # ul_harq_1
                1000,  # ul_harq_2
                1000   # ul_harq_3
            ])

            fixed_min_values = np.array([
                -140,  # ss_rsrp
                -20,   # ss_rsrq
                -20,   # ss_sinr
                1,     # ri
                0,     # wb_cqi_1tb
                0,     # cri
                -23,   # phr
                -20,   # pusch_snr
                0,     # ul_mcs1
                0,     # ul_bler
                0,     # ul_harq_0 (min difference)
                0,     # ul_harq_1
                0,     # ul_harq_2
                0      # ul_harq_3
            ])

            # Step 3: Min-Max normalization with fixed values
            range_vals = fixed_max_values - fixed_min_values
            range_vals[range_vals == 0] = 1  # Prevent division by zero

            normalized_matching_mac = (modified_matching_mac - fixed_min_values) / range_vals
            # print(normalized_matching_mac)

            matching_fh = fh_df[fh_df['TsTaiNs'] == TsTaiNs]
            # print(len(matching_fh))
            if len(matching_fh) != 1:
                continue
            # print(matching_fh)

            fh_samp = np.array(matching_fh['fhData'].iloc[0], dtype=np.float32)
            rx_slot = np.swapaxes(fh_samp.view(np.complex64).reshape(4, 14, 273*12), 2, 0)[:,:,0]
            # print(rx_slot)
            # print(rx_slot.shape)

            # rx_slot[fapi.rbStart: fapi.rbSize * 12, fapi.StartSymbolIndex : 14] = 0
            I = np.real(rx_slot)  # Real part
            Q = np.imag(rx_slot)  # Imaginary part
            IQ_tensor = np.stack([I, Q], axis=0)
            # print(IQ_tensor.shape) # (2, 3276, 14)

            # # fig, axs = plt.subplots(1)
            # # axs.imshow(10*np.log10(np.abs(rx_slot**2)), aspect='auto')
            # # axs.set_ylim([0, 273 * 12])
            # # axs.set_xlim([0, 14])
            # # axs.set_title('Ant ' + str(0))
            # # axs.set(xlabel='Symbol', ylabel='Resource Element')
            # # axs.label_outer()
            # # fig.suptitle('Power in RU Antennas') 
            # # plt.show(fig)
            
            matching_wgn = wgn_df[wgn_df['timestamp'] <= TsTaiNs]
            matching_wgn = matching_wgn.sort_values(by='timestamp', ascending=False).head(1)
            # print(matching_wgn['target_throughput_mbps'].to_numpy()[0])
            target_throughput_mbps = matching_wgn['target_throughput_mbps'].to_numpy()[0]
            normalization_target_throughput_mbps = target_throughput_mbps

            new_row = pd.DataFrame({
                'AdjDistanceDb': [-1], # Constant but unknown
                'NoiseLevelDb': [noise_level],
                'TargetThroughputMbps': [normalization_target_throughput_mbps],
                'MaxThroughputMbps': [throughput],
                'mac': [normalized_matching_mac],
                'fapi': [[fapi.rbStart/273, fapi.rbSize/273, fapi.StartSymbolIndex/14, 14/14]],
                'iq': [IQ_tensor],
            })
            dataset = pd.concat([dataset, new_row], ignore_index=True)
        #     break
        # break

    return dataset

def plot_dataset(dataset):
    import matplotlib.pyplot as plt

    unique_distances = np.sort(dataset['AdjDistanceDb'].unique())
    print(f'unique_distances: {unique_distances}')

    for distance in unique_distances:
        # plt.figure(figsize=(10, 6))
        distance_data = dataset[dataset['AdjDistanceDb'] == distance]

        plt.scatter(distance_data['NoiseLevelDb'], distance_data['MaxThroughputMbps'])

        for x, y in zip(distance_data['NoiseLevelDb'], distance_data['MaxThroughputMbps']):
            plt.plot([x, x], [0, y], 'k--', linewidth=0.5)
            plt.plot([-100, x], [y, y], 'k--', linewidth=0.5)

        plt.xlabel('Noise Level (dB)')
        plt.ylabel('Max Throughput (Mbps)')
        plt.title(f'AdjDistanceDb = {distance}')
        plt.xscale('symlog')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xticks([-100, -90, -80, -70, -60, -50, -40, -30])
        plt.ylim(0, 60)
        plt.show()

        target_throughput_mbps_s = np.stack(distance_data['TargetThroughputMbps'].to_numpy())
        max_throughput_mbps_s = np.stack(distance_data['MaxThroughputMbps'].to_numpy())
        plt.plot(target_throughput_mbps_s)
        plt.plot(max_throughput_mbps_s)
        plt.legend(['target_throughput_mbps', 'max_throughput_mbps'])
        plt.show()

def plot_metrics(dataset):
    plot_metrics = {
        'noise_level_s': np.stack(dataset['NoiseLevelDb'].to_numpy()),
        'target_throughput_mbps_s': np.stack(dataset['TargetThroughputMbps'].to_numpy()),
        'max_throughput_mbps_s': np.stack(dataset['MaxThroughputMbps'].to_numpy()),
        'ss_rsrp_s': np.average(np.stack(dataset['mac'])[:,:,0], axis=1),
        'ss_rsrq_s': np.average(np.stack(dataset['mac'])[:,:,1], axis=1),
        'ss_sinr_s': np.average(np.stack(dataset['mac'])[:,:,2], axis=1),
        'ri_s': np.average(np.stack(dataset['mac'])[:,:,3], axis=1),
        'wb_cqi_1tb_s': np.average(np.stack(dataset['mac'])[:,:,4], axis=1),
        'cri_s': np.average(np.stack(dataset['mac'])[:,:,5], axis=1),
        'phr_s': np.average(np.stack(dataset['mac'])[:,:,6], axis=1),
        'pusch_snr_s': np.average(np.stack(dataset['mac'])[:,:,7], axis=1),
        'ul_mcs1_s': np.average(np.stack(dataset['mac'])[:,:,8], axis=1),
        'ul_bler_s': np.average(np.stack(dataset['mac'])[:,:,9], axis=1),
        'ul_harq_0_s': np.average(np.stack(dataset['mac'])[:,:,10], axis=1),
        'ul_harq_1_s': np.average(np.stack(dataset['mac'])[:,:,11], axis=1),
        'ul_harq_2_s': np.average(np.stack(dataset['mac'])[:,:,12], axis=1),
        'ul_harq_3_s': np.average(np.stack(dataset['mac'])[:,:,13], axis=1),
    }
    fig, axs = plt.subplots(len(plot_metrics), 1, figsize=(12, 3*len(plot_metrics)))
    for i, (key, value) in enumerate(plot_metrics.items()):
        axs[i].plot(value, linewidth=1.5)
        axs[i].set_title(key, fontsize=10, pad=5)
        axs[i].set_xlabel('Time Step', fontsize=8)
        axs[i].set_ylabel(key, fontsize=8)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        axs[i].tick_params(axis='both', which='minor', labelsize=6)
    plt.tight_layout()
    plt.show()

def dataset_to_torch(dataset):
    import torch

    print("--- X ---")
    X_mac_torch = torch.tensor(np.stack(dataset['mac'].to_numpy()), dtype=torch.float32)
    print(f'X_mac_torch.shape: {X_mac_torch.shape}')
    X_fapi_torch = torch.tensor(np.stack(dataset['fapi'].to_numpy()), dtype=torch.float32)
    print(f'X_fapi_torch.shape: {X_fapi_torch.shape}')
    X_iq_torch = torch.tensor(np.stack(dataset['iq'].to_numpy()), dtype=torch.float32)
    print(f'X_iq_torch.shape: {X_iq_torch.shape}')

    print("--- Y ---")
    Y_noise_level_torch = torch.tensor(dataset['NoiseLevelDb'].to_numpy(), dtype=torch.float32)
    print(f'Y_noise_level_torch.shape: {Y_noise_level_torch.shape}')
    Y_target_throughput_mbps_torch = torch.tensor(dataset['TargetThroughputMbps'].to_numpy(), dtype=torch.float32)
    print(f'Y_target_throughput_mbps_torch.shape: {Y_target_throughput_mbps_torch.shape}')
    Y_max_throughput_mbps_torch = torch.tensor(dataset['MaxThroughputMbps'].to_numpy(), dtype=torch.float32)
    print(f'Y_max_throughput_mbps_torch.shape: {Y_max_throughput_mbps_torch.shape}')

    return X_mac_torch, X_fapi_torch, X_iq_torch, Y_noise_level_torch, Y_target_throughput_mbps_torch, Y_max_throughput_mbps_torch

def dataset_to_train_test_split(dataset):
    import torch
    from torch.utils.data import TensorDataset, Subset, DataLoader

    X_mac_torch, X_fapi_torch, X_iq_torch, Y_noise_level_torch, Y_target_throughput_mbps_torch, Y_max_throughput_mbps_torch = dataset_to_torch(dataset)
    
    # Set fixed seed for reproducibility
    seed = 100
    torch.manual_seed(seed)

    # Create TensorDataset
    tdataset = TensorDataset(X_mac_torch, X_fapi_torch, X_iq_torch, Y_noise_level_torch, Y_target_throughput_mbps_torch, Y_max_throughput_mbps_torch)

    # Size setup
    dataset_size = len(tdataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # Generate a random permutation of indices
    indices = torch.randperm(dataset_size).tolist()

    # Split indices into train and test
    train_indices = sorted(indices[:train_size])  # Take first train_size indices and sort
    test_indices = sorted(indices[train_size:])   # Take remaining indices and sort

    # Create Subsets
    train_dataset = Subset(tdataset, train_indices)
    test_dataset = Subset(tdataset, test_indices)

    print(f'train_dataset: {len(train_dataset)}')
    print(f'test_dataset: {len(test_dataset)}')

    return train_dataset, test_dataset

def xgboost_get_train_test(train_dataset, test_dataset, no_features):
    df_xgb_test = pd.DataFrame()
    df_xgb_train = pd.DataFrame()

    # Extract training data
    for i in range(len(train_dataset)):
        mac_data, fapi_data, iq_data, noise_level, target_throughput, max_throughput = train_dataset[i]
        df_xgb_train = pd.concat([df_xgb_train, pd.DataFrame({
            'mac': [np.array(mac_data)[-1,:no_features]],
            'noise_level': [np.array(noise_level)],
            'target_throughput': [np.array(target_throughput)],
            'y_max_throughput': [np.array(max_throughput)],
        })], ignore_index=True)

    # Extract test data
    for i in range(len(test_dataset)):
        mac_data, fapi_data, iq_data, noise_level, target_throughput, max_throughput = test_dataset[i]
        df_xgb_test = pd.concat([df_xgb_test, pd.DataFrame({
            'mac': [np.array(mac_data)[-1,:no_features]],
            'noise_level': [np.array(noise_level)],
            'target_throughput': [np.array(target_throughput)],
            'y_max_throughput': [np.array(max_throughput)],
        })], ignore_index=True)

    print(df_xgb_train)
    print(df_xgb_test)
    return df_xgb_train, df_xgb_test

def xgb_train(dtrain, dtest):
    import xgboost as xgb

    params = {
        'objective': 'reg:squarederror',  # Regression with squared error (MSE)
        'n_estimators': 100,             # Number of boosting rounds
        'max_depth': 19,                 # Maximum tree depth
        'reg_alpha': 1.2,                # L1 regularization
        'reg_lambda': 1.2,               # L2 regularization
        'subsample': 0.8,                # Fraction of samples per tree
        'min_child_weight': 3,           # Minimum sum of instance weight
        'colsample_bytree': 0.9,         # Fraction of features per tree
        'learning_rate': 0.1,            # Step size shrinkage (eta)
        'random_state': 42,              # For reproducibility
        'n_jobs': -1                     # Use all CPU cores
    }

    # Train the model with early stopping
    num_rounds = 200
    evals = [(dtrain, 'train'), (dtest, 'test')]
    xgb_model = xgb.train(params, dtrain, num_rounds, evals=evals, early_stopping_rounds=10, verbose_eval=False)

    return xgb_model

def xgb_test(xgb_model, dtest, df_test_model_xgb):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Make predictions
    p_max_throughput_mbps_xgb = xgb_model.predict(dtest)
    df_test_model_xgb['p_max_throughput'] = p_max_throughput_mbps_xgb

    # Calculate metrics
    r2 = r2_score(df_test_model_xgb['p_max_throughput'].to_numpy(), df_test_model_xgb['y_max_throughput'].to_numpy())
    mae = mean_absolute_error(df_test_model_xgb['p_max_throughput'].to_numpy(), df_test_model_xgb['y_max_throughput'].to_numpy())
    rmse = np.sqrt(mean_squared_error(df_test_model_xgb['p_max_throughput'].to_numpy(), df_test_model_xgb['y_max_throughput'].to_numpy()))

    print(f"XGBoost Model Results:")
    print(f"R2 Score: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

def xgb_plot(xgb_model, feature_names, df_test_model_xgb):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get feature importance scores (weight)
    importance = xgb_model.get_score(importance_type='weight')

    # Create a complete importance dictionary with all features
    complete_importance = {feat: importance.get(feat, 0.0) for feat in feature_names}
    print("Complete Importance (including zero scores):", complete_importance)

    # Sort features by importance (descending), include all 14
    sorted_importance = sorted(complete_importance.items(), key=lambda x: x[1], reverse=True)
    print("Number of features:", len(sorted_importance))

    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])

    # Plot feature importance (weight) with values at the end of bars
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='skyblue')
    plt.xlabel('Importance Score (Weight)')
    plt.title('Feature Importance (Weight)')
    plt.tight_layout()

    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 50, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                ha='left', va='center', fontsize=10)

    plt.show()

    # Get gain-based importance
    importance_gain = xgb_model.get_score(importance_type='gain')
    complete_importance_gain = {feat: importance_gain.get(feat, 0.0) for feat in feature_names}
    print("Complete Gain-based Importance:", complete_importance_gain)

    # Sort and convert to DataFrame
    sorted_importance_gain = sorted(complete_importance_gain.items(), key=lambda x: x[1], reverse=True)
    importance_gain_df = pd.DataFrame(sorted_importance_gain, columns=['Feature', 'Importance'])

    # Plot feature importance (gain) with values at the end of bars
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_gain_df['Feature'][::-1], importance_gain_df['Importance'][::-1], color='lightgreen')
    plt.xlabel('Importance Score (Gain)')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()

    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        max_gain = max(importance_gain_df['Importance'])
        offset = 0.05 * max_gain if max_gain > 0 else 0.1  # Avoid division by zero
        plt.text(width + offset, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center', fontsize=10)

    plt.show()

    # Create a more detailed time series plot with additional context
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Throughput comparison
    ax1.plot(df_test_model_xgb['y_max_throughput'].to_numpy(), label='Actual Throughput', color='blue', alpha=0.7)
    ax1.scatter(range(len(df_test_model_xgb['p_max_throughput'].to_numpy())), df_test_model_xgb['p_max_throughput'].to_numpy(), label='Predicted Throughput', color='red', alpha=0.7, s=1)
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.set_title('Throughput Prediction Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Prediction error
    error = df_test_model_xgb['p_max_throughput'].to_numpy() - df_test_model_xgb['y_max_throughput'].to_numpy()
    ax2.plot(error, color='purple', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Prediction Error (Mbps)')
    ax2.set_title('Prediction Error Over Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()