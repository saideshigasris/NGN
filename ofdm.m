%% ae_ofdm_end2end.m
% End-to-end Autoencoder replacing OFDM modulator+demodulator
% Requires: Deep Learning Toolbox, Communications Toolbox (for qammod/qamdemod, awgn)
clear; close all; clc;
rng(0);

%% PARAMETERS
Nsub = 64;            % OFDM subcarriers (encoder will produce Nsub complex samples)
M = 4;                % baseline QPSK
k = log2(M);          % bits per QAM symbol (2)
bitsPerFrame = Nsub * k;   % bits per AE frame (we train frame-level)
numTrainFrames = 8000;     % training frames
numValFrames   = 2000;     % validation frames
batchSize = 128;
numEpochs = 40;
trainSNR_dB = 8;      % SNR used during training (or choose random per batch)
snrRange = 0:2:20;    % SNRs for evaluation

learningRate = 1e-3;

fprintf('AE-OFDM: Nsub=%d bits/frame=%d trainFrames=%d valFrames=%d\n',...
    Nsub, bitsPerFrame, numTrainFrames, numValFrames);

%% -------------------------
% Create data (bits) for training/validation and for OFDM baseline
%% -------------------------
% Training bits (frames)
Xtrain_bits = randi([0 1], bitsPerFrame, numTrainFrames);   % frameLen x nTrain
Xval_bits   = randi([0 1], bitsPerFrame, numValFrames);

% Test set (for BER vs SNR)
numTestFrames = 2000;
Xtest_bits = randi([0 1], bitsPerFrame, numTestFrames);

%% -------------------------
% Build encoder & decoder as dlnetwork objects
% Encoder: bits -> dense -> relu -> dense -> linear outputs (2*Nsub real outputs)
% Decoder: 2*Nsub real inputs -> dense -> relu -> dense -> sigmoid outputs (bits)
%% -------------------------
encoderLayers = [
    featureInputLayer(bitsPerFrame,'Name','bits_in','Normalization','none')
    fullyConnectedLayer(512,'Name','fc1_enc')
    reluLayer('Name','relu1_enc')
    fullyConnectedLayer(256,'Name','fc2_enc')
    reluLayer('Name','relu2_enc')
    fullyConnectedLayer(2*Nsub,'Name','fc_out_enc') % real+imag interleaved: [Re(1..N); Im(1..N)]
    % normalization (power) will be done in forward function
    ];

decoderLayers = [
    featureInputLayer(2*Nsub,'Name','rx_in','Normalization','none')
    fullyConnectedLayer(256,'Name','fc1_dec')
    reluLayer('Name','relu1_dec')
    fullyConnectedLayer(512,'Name','fc2_dec')
    reluLayer('Name','relu2_dec')
    fullyConnectedLayer(bitsPerFrame,'Name','fc_out_dec')
    sigmoidLayer('Name','sig_out')
    ];

% Convert to layerGraph and dlnetwork
encLG = layerGraph(encoderLayers);
decLG = layerGraph(decoderLayers);

encNet = dlnetwork(encLG);
decNet = dlnetwork(decLG);

%% -------------------------
% Training setup (Adam)
%% -------------------------
% Adam state
avgGradEnc = []; avgSqGradEnc = [];
avgGradDec = []; avgSqGradDec = [];

trailingAvgEnc = []; trailingAvgSqEnc = [];
trailingAvgDec = []; trailingAvgSqDec = [];

% Helper: binary cross-entropy loss
bceLoss = @(yPred, yTrue) -mean( yTrue .* log(yPred+1e-9) + (1-yTrue).*log(1-yPred+1e-9) , 'all');

% Mini-batch partitions
numIterationsPerEpoch = floor(numTrainFrames / batchSize);

%% -------------------------
% Training loop
%% -------------------------
fprintf('Training AE end-to-end (epochs=%d, batchsize=%d)...\n', numEpochs, batchSize);
iteration = 0;
for epoch = 1:numEpochs
    % shuffle training frames
    idx = randperm(numTrainFrames);
    Xtrain_bits = Xtrain_bits(:, idx);
    
    epochLoss = 0;
    for it = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        batchIdx = (it-1)*batchSize + (1:batchSize);
        bitsBatch = Xtrain_bits(:, batchIdx); % bitsPerFrame x batchSize
        
        % Convert bits -> double (0/1)
        bitsBatchDL = dlarray(single(bitsBatch),'CB'); % C x B where C=bitsPerFrame
        
        % Forward + loss and gradients
        % Use dlfeval to enable automatic differentiation
        [loss, gradientsEnc, gradientsDec] = dlfeval(@modelLoss, encNet, decNet, bitsBatchDL, trainSNR_dB, bitsPerFrame, Nsub, bceLoss);
        epochLoss = epochLoss + double(gather(extractdata(loss)));
        
        % Update encoder parameters (Adam)
        [encNet, trailingAvgEnc, trailingAvgSqEnc] = adamupdate(encNet, gradientsEnc, trailingAvgEnc, trailingAvgSqEnc, iteration, learningRate);
        % Update decoder parameters
        [decNet, trailingAvgDec, trailingAvgSqDec] = adamupdate(decNet, gradientsDec, trailingAvgDec, trailingAvgSqDec, iteration, learningRate);
    end
    
    epochLoss = epochLoss / numIterationsPerEpoch;
    
    % Validation loss (one forward on validation set)
    valBitsDL = dlarray(single(Xval_bits(:,1:batchSize)),'CB');
    valPred = forwardAE(encNet, decNet, valBitsDL, trainSNR_dB, bitsPerFrame, Nsub);
    valLoss = bceLoss(valPred, double(Xval_bits(:,1:batchSize)));
    
    fprintf('Epoch %2d/%d  TrainLoss=%.4f  ValLoss=%.4f\n', epoch, numEpochs, epochLoss, double(gather(extractdata(valLoss))));
end

%% -------------------------
% EVALUATION: BER vs SNR for AE and Traditional OFDM baseline
%% -------------------------
fprintf('Evaluating BER over SNR range...\n');
berAE = zeros(size(snrRange));
berOFDM = zeros(size(snrRange));

% Prepare OFDM baseline transmit signal (one long sequence built from Xtest_bits)
% Map bits->QPSK symbols , form OFDM frames etc. We'll evaluate OFDM per SNR below for fairness.
% Precompute OFDM tx frames per test bits:
% Convert bits -> symbol ints
bitsTest = Xtest_bits(:);
symbols_reshaped = reshape(bitsTest, k, []).';
symbols_int = zeros(size(symbols_reshaped,1),1);
for i=1:size(symbols_reshaped,1)
    % left-msb mapping
    val = 0;
    for b=1:k
        val = val*2 + symbols_reshaped(i,b);
    end
    symbols_int(i) = val;
end
% QPSK modulation (unit avg power)
tx_symbols = qammod(symbols_int, M, 'InputType','integer','UnitAveragePower',true);
tx_symMatrix = reshape(tx_symbols, Nsub, []);   % Nsub x numFramesTest
tx_ofdm_time = ifft(tx_symMatrix, Nsub, 1);
% add CP
cpLen = 16;
tx_ofdm_cp = [tx_ofdm_time(end-cpLen+1:end,:); tx_ofdm_time];
tx_ofdm_stream = tx_ofdm_cp(:);

% For AE evaluation we will process frames individually
numTestFrames = size(Xtest_bits,2);

for sIdx = 1:length(snrRange)
    snr = snrRange(sIdx);
    
    % ----- AE path -----
    % Feed all test frames through encoder, channel, decoder
    bits_test_dl = dlarray(single(Xtest_bits),'CB');
    predProb = forwardAE(encNet, decNet, bits_test_dl, snr, bitsPerFrame, Nsub); % outputs probabilities
    predBits = gather(extractdata(predProb > 0.5));
    % predBits is bitsPerFrame x numTestFrames
    AE_bits_flat = predBits(:);
    AE_true_bits_flat = Xtest_bits(:);
    berAE(sIdx) = mean(AE_bits_flat ~= AE_true_bits_flat);
    
    % ----- OFDM baseline -----
    % AWGN on tx_ofdm_stream
    rx_ofdm = awgn(tx_ofdm_stream, snr, 'measured');
    % reshape back
    rx_cp = reshape(rx_ofdm, Nsub+cpLen, []);
    rx_no_cp = rx_cp(cpLen+1:end,:);
    rx_fft = fft(rx_no_cp, Nsub, 1);
    rx_symbols_vec = rx_fft(:);
    rx_ints = qamdemod(rx_symbols_vec, M, 'OutputType','integer','UnitAveragePower',true);
    % convert ints back to bits (left-msb)
    rx_bits_matrix = de2bi(rx_ints, k, 'left-msb');  % requires Communications Toolbox / or custom
    rx_bits = rx_bits_matrix.'; rx_bits = rx_bits(:);
    berOFDM(sIdx) = mean(rx_bits ~= bitsTest);
    
    fprintf('SNR=%2d dB  AE-BER=%.3e  OFDM-BER=%.3e\n', snr, berAE(sIdx), berOFDM(sIdx));
end

%% -------------------------
% Plot BER
%% -------------------------
figure('Color','w'); semilogy(snrRange, berOFDM, 'bo-','LineWidth',2); hold on;
semilogy(snrRange, berAE, 'r*-','LineWidth',2);
grid on; xlabel('SNR (dB)'); ylabel('BER'); title('BER: AE-OFDM vs Traditional OFDM');
legend('Traditional OFDM (QPSK+IFFT)','AE-OFDM (learned end-to-end)','Location','southwest');

%% -------------------------
% Supporting functions (subfunctions)
%% -------------------------
function [loss, gradientsEnc, gradientsDec] = modelLoss(encNet, decNet, bitsBatchDL, snr_dB, bitsPerFrame, Nsub, bceLoss)
    % bitsBatchDL: C x B (bits per frame x batch size), values 0/1 (single/dlarray)
    % Forward encoder
    encOut = forward(encNet, bitsBatchDL); % (2*Nsub) x B real outputs
    % Normalize encoder power per batch (so average transmit power = 1)
    encOutData = encOut; % dlarray
    % reshape to [2,Nsub,B] to compute power
    % encOut order: [2*Nsub x B], first Nsub = real parts, next Nsub = imag parts
    % compute power across all elements and normalize
    power = mean(encOutData.^2, 'all');
    encOutNorm = encOutData ./ sqrt(power + 1e-9);
    % form complex samples: Re = first Nsub rows, Im = next Nsub rows
    Re = encOutNorm(1:Nsub,:);
    Im = encOutNorm(Nsub+1:end,:);
    % combine to complex for channel simulation (we'll add noise to real+imag separately)
    % Convert to numeric for AWGN addition (AWGN expects numeric arrays)
    ReNum = gather(extractdata(Re));
    ImNum = gather(extractdata(Im));
    % AWGN per sample (assume 'measured' uses signal power) -> compute noise
    txComplex = complex(ReNum, ImNum); % Nsub x batch
    % Now add AWGN with given snr_dB
    rxComplex = awgn(txComplex, snr_dB, 'measured');
    % Convert back to real concatenated
    rxReal = real(rxComplex);
    rxImag = imag(rxComplex);
    rxConcat = [rxReal; rxImag]; % (2*Nsub) x batch (numeric)
    rxDL = dlarray(single(rxConcat),'CB');
    % Forward decoder
    preds = forward(decNet, rxDL); % bitsPerFrame x batch, values in (0,1)
    % Compute loss (binary cross-entropy)
    loss = bceLoss(preds, double(bitsBatchDL));
    % Compute gradients
    [gradEnc, gradDec] = dlgradient(loss, encNet.Learnables.Value, decNet.Learnables.Value, 'RetainData', true);
    % Pack gradients in table-like form accepted by adamupdate wrapper below
    % Convert gradient cell arrays to same structure as networks' Learnables
    gradientsEnc = gradEnc;
    gradientsDec = gradDec;
end

function yPred = forwardAE(encNet, decNet, bitsDL, snr_dB, bitsPerFrame, Nsub)
    % Forward path used for validation/test (vectorized)
    encOut = forward(encNet, bitsDL); % 2*Nsub x B
    % normalize
    power = mean(encOut.^2, 'all');
    encOutNorm = encOut ./ sqrt(power + 1e-9);
    % build complex
    Re = encOutNorm(1:Nsub,:);
    Im = encOutNorm(Nsub+1:end,:);
    ReNum = gather(extractdata(Re));
    ImNum = gather(extractdata(Im));
    txComplex = complex(ReNum, ImNum);
    rxComplex = awgn(txComplex, snr_dB, 'measured');
    rxConcat = [real(rxComplex); imag(rxComplex)];
    rxDL = dlarray(single(rxConcat),'CB');
    preds = forward(decNet, rxDL);
    yPred = preds;
end

function [netUpdated, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteration, learnRate)
    % Very small helper to update dlnetwork parameters using gradients (structure)
    % gradients and net.Learnables are corresponding tables. We'll update net.Learnables.Value entries.
    % This implementation uses simple symbolics: we extract gradients cell/array and update values
    % Note: MATLAB doesn't provide a single built-in 'adamupdate' for dlnetwork; use this helper.
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
    if isempty(trailingAvg)
        trailingAvg = cellfun(@(x) zeros(size(x),'like',x), net.Learnables.Value, 'UniformOutput', false);
        trailingAvgSq = cellfun(@(x) zeros(size(x),'like',x), net.Learnables.Value, 'UniformOutput', false);
    end
    % Iterate over learnable parameters
    for i = 1:numel(net.Learnables.Value)
        g = gradients{i};    % gradient (dlarray -> numeric)
        if isa(g,'dlarray')
            g = extractdata(g);
        end
        % convert net parameter to numeric
        param = net.Learnables.Value{i};
        % update moving averages
        trailingAvg{i} = beta1 .* trailingAvg{i} + (1-beta1) .* g;
        trailingAvgSq{i} = beta2 .* trailingAvgSq{i} + (1-beta2) .* (g.^2);
        % bias-corrected
        corrAvg = trailingAvg{i} ./ (1 - beta1^iteration);
        corrAvgSq = trailingAvgSq{i} ./ (1 - beta2^iteration);
        % parameter update
        update = learnRate .* corrAvg ./ (sqrt(corrAvgSq) + epsilon);
        % apply update (net.Learnables.Value stored as arrays; assign)
        net.Learnables.Value{i} = param - update;
    end
    netUpdated = net;
end