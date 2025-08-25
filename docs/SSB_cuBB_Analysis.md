# SSB (Synchronization Signal Block) Analysis in cuBB System

## Overview

SSB (Synchronization Signal Block) is a critical component in 5G NR that consists of four main elements:
- **PSS** (Primary Synchronization Signal)
- **SSS** (Secondary Synchronization Signal) 
- **PBCH** (Physical Broadcast Channel)
- **DMRS** (Demodulation Reference Signal)

## SSB Structure

### **📊 SSB Block Configuration**
```
SSB spans 4 OFDM symbols and 240 subcarriers:
┌─────────────────────────────────────────────────────────────┐
│ Symbol 0: PSS (subcarriers 56-182)                         │
│ Symbol 1: PBCH + DMRS                                      │
│ Symbol 2: SSS (subcarriers 56-182)                         │
│ Symbol 3: PBCH + DMRS                                      │
└─────────────────────────────────────────────────────────────┘
```

### **🔧 Resource Mapping**
- **PSS**: Subcarriers 56-182 in Symbol 0
- **SSS**: Subcarriers 56-182 in Symbol 2  
- **PBCH**: Subcarriers 0-239 in Symbols 1,3
- **DMRS**: Embedded within PBCH symbols

## PSS (Primary Synchronization Signal)

### **🔧 Implementation in cuBB**

#### **A. Sequence Generation**
```cpp
// From ss.cu:336
scalar_t tmpSS0 = static_cast<scalar_t>(beta_pss * (1 - 2*SSB_PSS_X_EXT[idxPssSss + 43*NID2]));
```

#### **B. Sequence Table**
```cpp
// From ss.cu:42-43
static __device__ __constant__ uint8_t SSB_PSS_X_EXT[] = {
    0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1
    // ... extended sequence to avoid modulo operations
};
```

#### **C. PSS Value Range**
```cpp
// PSS uses BPSK modulation with custom scaling
// Sequence values: 0 or 1
// Final PSS values: (1 - 2*bit) * beta_pss
// Range: [-beta_pss, +beta_pss]

// Example with beta_pss = 1.0:
// Min PSS Value: -1.0
// Max PSS Value: +1.0
```

### **🔧 Power Control (beta_pss)**

#### **A. Configuration Values**
| FAPI Value | dB Increase | Linear Value | Description |
|------------|-------------|--------------|-------------|
| **0** | 0 dB | 1.0 | Normal PSS power |
| **1** | 3 dB | 1.4125 | 3 dB power boost |

#### **B. Implementation**
```cpp
// From scf_5g_slot_commands.cpp:747-762
switch (cmd.beta_pss) {
    case 0:
        block_params.beta_pss = 1;                    // 0 dB power increase
        break;
    case 1:
        block_params.beta_pss = std::pow(10.0, 3.0 / 20.0);  // 3 dB power increase
        break;
    default:
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Unknown value for betaPss: {}", cmd.beta_pss);
        break;
}
```

## SSS (Secondary Synchronization Signal)

### **🔧 Implementation in cuBB**

#### **A. Sequence Generation**
```cpp
// From ss.cu:350-355
int16_t lower_half = 1 - 2*SSB_SSS_X0[(idxPssSss + m0) % CUPHY_SSB_N_SS_SEQ_BITS];
int16_t upper_half = 1 - 2*SSB_SSS_X1[(idxPssSss + m1) % CUPHY_SSB_N_SS_SEQ_BITS];
scalar_t tmpSS1 = static_cast<scalar_t>(beta_sss * (lower_half * upper_half));
```

#### **B. SSS Value Range**
```cpp
// SSS uses QPSK-like modulation
// Range: [-beta_sss, 0, +beta_sss]
// Values: -1, 0, +1 (multiplied by beta_sss)
```

### **🔧 Power Control (beta_sss)**
```cpp
// From scf_5g_slot_commands.cpp:762
block_params.beta_sss = 1;  // SSS always uses 1.0
```

## PBCH (Physical Broadcast Channel)

### **🔧 Implementation in cuBB**

#### **A. PBCH Processing Pipeline**
```cpp
// From ssb_tx.cpp:306-433
// 1. Payload generation with CRC
// 2. Polar encoding
// 3. Rate matching
// 4. Scrambling
// 5. QPSK modulation
// 6. Resource mapping
```

#### **B. PBCH Power Control**
```cpp
// From ss.cu:500-510
qam.x = static_cast<scalar_t>(beta_sss_factor * (1 - 2 * x));
qam.y = static_cast<scalar_t>(beta_sss_factor * (1 - 2 * y));
// Where beta_sss_factor = beta_sss * 0.70710678f
```

## DMRS (Demodulation Reference Signal)

### **🔧 Implementation in cuBB**

#### **A. DMRS Generation**
```cpp
// From ss.cu:514-520
// Gold sequence generation for DMRS
uint32_t c_init = (0x1 << 11) * (i_ssb + 1) * (NID / 4 + 1) +
                  (0x1 << 6) * (i_ssb + 1) + NID % 4;
dmrs_seq[tid] = gold32(c_init, tid * 32);
```

#### **B. DMRS Power Control**
```cpp
// From scf_5g_slot_commands.cpp:509
dci.beta_dmrs = std::pow(10.0, (pwr_info.power_control_offset_ss_profile_nr/20.0));
```

## Power Control Summary

### **🔧 Power Scaling Factors**

| Signal | Power Factor | Default Value | Purpose |
|--------|--------------|---------------|---------|
| **PSS** | `beta_pss` | 1.0 or 1.4125 | Synchronization detection |
| **SSS** | `beta_sss` | 1.0 | Cell identification |
| **PBCH** | `beta_sss * 0.707` | 0.707 | Broadcast information |
| **DMRS** | `beta_dmrs` | Configurable | Channel estimation |

### **🔧 Power Relationships**
```cpp
// PSS can have higher power than SSS for better detection
// PBCH uses reduced power (0.707 factor) for power efficiency
// DMRS power is independently configurable
```

## SSB Configuration Parameters

### **🔧 Cell-Level Parameters**
```cpp
// From cuphy_api.h:3747
typedef struct _cuphyPerCellSsbDynPrms {
    uint16_t NID;        // Physical cell ID
    uint16_t SFN;        // System frame number
    uint8_t  nHF;        // Half frame index (0 or 1)
    uint8_t  Lmax;       // Max SS blocks (4, 8, or 64)
    uint16_t k_SSB;      // SSB subcarrier offset
} cuphyPerCellSsbDynPrms_t;
```

### **🔧 SSB-Level Parameters**
```cpp
// From cuphy_api.h:3723
typedef struct _cuphyPerSsBlockDynPrms {
    uint16_t f0;         // Initial SSB subcarrier index
    uint8_t  t0;         // Initial SSB OFDM symbol index
    uint8_t  blockIndex; // SS block index (0 - L_max)
    float    beta_pss;   // PSS power scaling factor
    float    beta_sss;   // SSS power scaling factor
    uint16_t cell_index; // Cell index
    uint8_t  enablePrcdBf; // Enable precoding
    uint16_t pmwPrmIdx;  // Precoding matrix index
} cuphyPerSsBlockDynPrms_t;
```

## Beamforming Support

### **🔧 Precoding Implementation**
```cpp
// From ss.cu:340-350
if(enablePrcdBf) {
    for(int idx = 0; idx < nPorts; idx++) {
        tf_signal[idxPssSss + 56 + offset_per_port*idx].x = tmpSS0 * d_pmw_params[pmwPrmIdx].matrix[idx].x;
        tf_signal[idxPssSss + 56 + offset_per_port*idx].y = tmpSS0 * d_pmw_params[pmwPrmIdx].matrix[idx].y;
    }
}
```

## Value Ranges Summary

### **🔧 Signal Value Ranges**

| Signal | Modulation | Value Range | Normalization |
|--------|------------|-------------|---------------|
| **PSS** | BPSK | [-beta_pss, +beta_pss] | Custom scaling |
| **SSS** | QPSK-like | [-beta_sss, 0, +beta_sss] | beta_sss = 1.0 |
| **PBCH** | QPSK | [-0.707, +0.707] | beta_sss * 0.707 |
| **DMRS** | QPSK | [-beta_dmrs, +beta_dmrs] | Configurable |

### **🔧 Power Relationships**
- **PSS**: Can be boosted by 3 dB for better detection
- **SSS**: Standard power level
- **PBCH**: Reduced power for efficiency
- **DMRS**: Independently configurable power

## Key Features

### **🔧 cuBB Optimizations**
1. **GPU-optimized**: CUDA kernels for parallel processing
2. **Memory efficient**: Pre-computed sequence tables
3. **Flexible power control**: Independent scaling for each signal
4. **Beamforming support**: Multi-antenna transmission
5. **Standards compliant**: Follows 3GPP specifications

### **🔧 Performance Benefits**
- **Fast synchronization**: Optimized PSS/SSS detection
- **Reliable broadcast**: Robust PBCH transmission
- **Efficient power usage**: Optimized power allocation
- **Scalable design**: Supports multiple cells and SSBs

## Conclusion

The SSB implementation in cuBB provides a comprehensive, GPU-optimized solution for 5G NR synchronization signals. It offers flexible power control, beamforming support, and efficient resource utilization while maintaining full compliance with 3GPP standards.
