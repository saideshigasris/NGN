Autoencoder Replacing the OFDM Modulator and Demodulator

A deep learning approach to creating an end-to-end communication system that replaces the traditional Orthogonal Frequency-Division Multiplexing (OFDM) modulation and demodulation blocks, with a focus on Bit Error Rate (BER) performance under AWGN channel conditions111.

 Aim & Overview
The primary goal is to develop an end-to-end Autoencoder-based OFDM system (AE-OFDM) that replaces the traditional OFDM modulator/demodulator chain using deep learning and compare its BER performance with traditional OFDM under AWGN channel conditions2.
AE-OFDM Concept 3
The Autoencoder (AE) is trained to replace the entire traditional physical layer block4:
Traditional OFDM Component	Replaced by Autoencoder	Functionality
Traditional Chain	QPSK $\rightarrow$ IFFT $\rightarrow$ CP $\rightarrow$ CHANNEL $\rightarrow$ Remove CP $\rightarrow$ FFT $\rightarrow$ Demodulation 5	
Encoder NN	Modulator + IFFT + Mapping 6	Converts input bits into 64 complex OFDM-like samples7.
Decoder NN	FFT + Equalizer + Demodulator 8	Reconstructs bits using neural layers after transmission through the AWGN channel9.
Why Autoencoder? 10
•	Learns the best waveform for the given channel11.
•	Aims for better BER at low SNR12.
•	Learns implicit equalization13.
•	Avoids strict constraints of FFT/IFFT14.

 System Architecture & Procedure
Key Parameters 15
Parameter	Value	Description
Subcarriers ($N_{\text{sub}}$)	64	OFDM subcarriers (complex samples)16.
Modulation ($M$)	4 (QPSK baseline) 17171717	Bits per QAM symbol $k = \log_2(M) = 2$18.
Bits per Frame	128 ($64 \times 2$) 19191919	Input length for the AE.
Training Frames	8,000 20202020	Number of frames used for training.
Epochs	40 21212121	Number of training iterations.
Training SNR	8 dB 2222	SNR used during the training phase.
Model Architecture (Layers)
The system uses fully connected (Dense) layers:
•	Encoder Network (Modulator):
o	Input: bitsPerFrame (128 bits)
o	Layers: $\text{Dense}(512) \rightarrow \text{ReLU} \rightarrow \text{Dense}(256) \rightarrow \text{ReLU} \rightarrow \text{Dense}(128)$ 23.
o	Output: $2 \times N_{\text{sub}} = 128$ real outputs (I/Q)24242424.
o	Post-processing: Normalize power25.
•	Decoder Network (Demodulator):
o	Input: $2 \times N_{\text{sub}} = 128$ real inputs (Received I/Q samples)26.
o	Layers: $\text{Dense}(256) \rightarrow \text{ReLU} \rightarrow \text{Dense}(512) \rightarrow \text{ReLU} \rightarrow \text{Dense}(128)$ 27.
o	Output Layer: Sigmoid activation28282828.
Training Procedure 29
The network is trained end-to-end using the following steps:
1.	Generate a batch of random input bits30.
2.	Pass through the Encoder, add AWGN at $8\text{ dB}$, and decode 31313131.
3.	Compute Binary Cross-Entropy (BCE) Loss32.
4.	Perform Backpropagation using the Adam update rule33333333.

Results and Comparison
The final performance evaluation compares the AE-OFDM system against the traditional OFDM baseline (QPSK + IFFT/FFT) across an SNR range of $0\text{ dB}$ to $20\text{ dB}$ 34343434.
BER Performance vs. SNR
The data shows the BER achieved by both systems35:
SNR (dB)	AE-OFDM BER	Traditional OFDM BER
0	$3.722 \times 10^{-1}$	$1.578 \times 10^{-1}$
8	$2.719 \times 10^{-1}$	$6.160 \times 10^{-3}$
12	$2.417 \times 10^{-1}$	$3.516 \times 10^{-5}$
20	$2.175 \times 10^{-1}$	$0.000 \times 10^{0}$
Visualization
The plot illustrates the performance gap.
•	The Traditional OFDM system shows a typical error floor, achieving near-zero BER above $14\text{ dB}$.
•	The AE-OFDM system demonstrates a significantly higher BER across all tested SNRs, though the prompt notes that the autoencoder performs better at low SNR in principle36. (The code output shows the Traditional OFDM baseline outperforming the AE in this particular experiment
 Technical Details
•	Programming Environment: MATLAB37.
•	Required Toolboxes: Deep Learning Toolbox and Communications Toolbox38.
•	Loss Function: Binary Cross-Entropy (BCE)39.
•	Optimizer: Adam with a learning rate of $1\times 10^{-3}$40404040.

