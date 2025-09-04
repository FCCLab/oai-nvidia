# IQ Range Summary for All Modulation Schemes in cuBB

## Modulation Scheme IQ Value Ranges

| Modulation | Normalization Factor (A) | Min I/Q Value | Max I/Q Value | Range Width | PAM Levels | Average Power |
|------------|-------------------------|---------------|---------------|-------------|------------|---------------|
| **BPSK** | 0.707106781186547 | -0.707106781186547 | +0.707106781186547 | 1.414213562373095 | ±1A | 1.0 |
| **QPSK** | 0.707106781186547 | -0.707106781186547 | +0.707106781186547 | 1.414213562373095 | ±1A | 1.0 |
| **QAM16** | 0.316227766016838 | -0.948683298050514 | +0.948683298050514 | 1.897366596101028 | ±1A, ±3A | 1.0 |
| **QAM64** | 0.154303349962092 | -1.080123449734644 | +1.080123449734644 | 2.160246899469288 | ±1A, ±3A, ±5A, ±7A | 1.0 |
| **QAM256** | 0.076696498884737 | -1.150447483271055 | +1.150447483271055 | 2.300894966542110 | ±1A, ±3A, ±5A, ±7A, ±9A, ±11A, ±13A, ±15A | 1.0 |

## Key Parameters

### Normalization Factors
- **BPSK/QPSK**: A = 1/√2 ≈ 0.7071
- **QAM16**: A = 1/√10 ≈ 0.3162
- **QAM64**: A = 1/√42 ≈ 0.1543
- **QAM256**: A = 1/√170 ≈ 0.0767

### Mathematical Formulas
```cpp
// BPSK: (1 - 2*b[0]) * A
// QPSK: (1 - 2*b[0]) + 1j*(1 - 2*b[1]) * A
// QAM16: (1 - 2*b[0]) * (2 - (1 - 2*b[2])) * A
// QAM64: (1 - 2*b[0]) * (4 - (1 - 2*b[2]) * (2 - (1 - 2*b[4]))) * A
// QAM256: (1 - 2*b[0]) * (8 - (1 - 2*b[2]) * (4 - (1 - 2*b[4]) * (2 - (1 - 2*b[6])))) * A
```

### Important Notes
- All schemes normalized for **average symbol power = 1.0**
- Higher-order QAM has larger peak values but same average power
- Ranges used in compression/decompression with `beta_dl` scaling
- Values ensure signals fit within compression format without saturation
