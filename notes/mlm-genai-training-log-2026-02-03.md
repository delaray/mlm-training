(mlm-training) C:\projects\mlm-training>python example_mlm_training.py
2026-02-03 10:54:26,778 - root - INFO - ================================================================================
2026-02-03 10:54:26,778 - root - INFO - MLM TRAINING PIPELINE
2026-02-03 10:54:26,778 - root - INFO - ================================================================================
2026-02-03 10:54:26,778 - root - INFO - Model: microsoft/deberta-v3-base
2026-02-03 10:54:26,778 - root - INFO - Data Directory: data/books
2026-02-03 10:54:26,778 - root - INFO - Output Directory: results\training-deberta-v3-base-20260203-1054
2026-02-03 10:54:26,778 - root - INFO - ================================================================================
2026-02-03 10:54:26,783 - root - INFO -
================================================================================
2026-02-03 10:54:26,783 - root - INFO - STEP 1: PREPARING DATASET
2026-02-03 10:54:26,783 - root - INFO - ================================================================================

2026-02-03 10:54:26,783 - root - INFO - ================================================================================
2026-02-03 10:54:26,783 - root - INFO - Starting MLM Dataset Preparation
2026-02-03 10:54:26,783 - root - INFO - ================================================================================
2026-02-03 10:54:26,783 - root - INFO - Loading tokenizer from: models\deberta-v3-base
2026-02-03 10:54:26,993 - root - INFO - Reading documents from: data/books
Document loader: <src.ingest.DocumentIngest object at 0x00000243D63F01A0>
Reading files in data/books
Error reading data/books\LLM-Twin-Project-Architecture--Decoding-ML.jpg, document not parsed
cannot access local variable 'text' where it is not associated with a value
2026-02-03 10:54:33,427 - root - INFO - Successfully read 18 files
2026-02-03 10:54:33,427 - root - INFO - Total chunks available: 3952
2026-02-03 10:54:33,481 - root - INFO - Splitting dataset (test_split=0.1)
2026-02-03 10:54:33,488 - root - INFO - Tokenizing dataset...
Tokenizing (num_proc=4): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3556/3556 [00:08<00:00, 436.98 examples/s]
Tokenizing (num_proc=4): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 396/396 [00:07<00:00, 52.38 examples/s]
2026-02-03 10:54:49,228 - root - INFO - Training samples: 3556
2026-02-03 10:54:49,228 - root - INFO - Test samples: 396
2026-02-03 10:54:49,228 - root - INFO - Max sequence length: 512
2026-02-03 10:54:49,228 - root - INFO - Dataset preparation completed in 0:00:22.445183
2026-02-03 10:54:49,228 - root - INFO - ================================================================================
2026-02-03 10:54:49,230 - root - INFO - ✓ Dataset prepared successfully
--- Logging error ---
Traceback (most recent call last):
  File "C:\Python313\Lib\logging\__init__.py", line 1153, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 40: character maps to <undefined>
Call stack:
  File "C:\projects\mlm-training\example_mlm_training.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\example_mlm_training.py", line 118, in main
    logging.info(f"✓ Dataset prepared successfully")
Message: '✓ Dataset prepared successfully'
Arguments: ()
2026-02-03 10:54:49,292 - root - INFO -   Training samples: 3556
2026-02-03 10:54:49,292 - root - INFO -   Test samples: 396
2026-02-03 10:54:49,292 - root - INFO -
================================================================================
2026-02-03 10:54:49,292 - root - INFO - STEP 2: SETTING UP MODEL
2026-02-03 10:54:49,292 - root - INFO - ================================================================================

2026-02-03 10:54:49,293 - root - INFO - ================================================================================
2026-02-03 10:54:49,293 - root - INFO - Setting up model for MLM training
2026-02-03 10:54:49,293 - root - INFO - ================================================================================
2026-02-03 10:54:49,293 - root - INFO - Loading model without quantization (on CPU first)
`torch_dtype` is deprecated! Use `dtype` instead!
2026-02-03 10:54:49,438 - root - INFO - Model loaded on CPU
2026-02-03 10:54:49,438 - root - INFO - Applying LoRA configuration on CPU
2026-02-03 10:54:49,438 - root - INFO - Auto-detected target modules: ['query', 'key', 'value', 'dense']
C:\projects\mlm-training\.venv\Lib\site-packages\torch\cuda\__init__.py:374: UserWarning: Found GPU0 NVIDIA GeForce RTX 5090 Laptop GPU which is of compute capability (CC) 12.0.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 5.0 which supports hardware CC >=5.0,<6.0 except {5.3}
- 6.0 which supports hardware CC >=6.0,<7.0 except {6.2}
- 6.1 which supports hardware CC >=6.1,<7.0 except {6.2}
- 7.0 which supports hardware CC >=7.0,<8.0 except {7.2}
- 7.5 which supports hardware CC >=7.5,<8.0
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 8.6 which supports hardware CC >=8.6,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
Please follow the instructions at https://pytorch.org/get-started/locally/ to install a PyTorch release that supports one of these CUDA versions: 12.8, 13.0
  _warn_unsupported_code(d, device_cc, code_ccs)
C:\projects\mlm-training\.venv\Lib\site-packages\torch\cuda\__init__.py:492: UserWarning:
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  queued_call()
W0203 10:54:50.480000 14400 .venv\Lib\site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
trainable params: 1,794,048 || all params: 186,345,828 || trainable%: 0.9628
2026-02-03 10:54:53,655 - root - INFO - LoRA applied successfully on CPU
2026-02-03 10:54:53,655 - root - INFO - Keeping model on CPU (RTX 5090 sm_120 lacks required CUDA kernels)
2026-02-03 10:54:53,655 - root - INFO - Note: Training will use CPU. For GPU training, wait for PyTorch with full sm_120 support.
2026-02-03 10:54:53,656 - root - INFO - ================================================================================
2026-02-03 10:54:53,656 - root - INFO - ✓ Model setup complete
--- Logging error ---
Traceback (most recent call last):
  File "C:\Python313\Lib\logging\__init__.py", line 1153, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 40: character maps to <undefined>
Call stack:
  File "C:\projects\mlm-training\example_mlm_training.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\example_mlm_training.py", line 141, in main
    logging.info(f"✓ Model setup complete")
Message: '✓ Model setup complete'
Arguments: ()
2026-02-03 10:54:53,656 - root - INFO -   Quantized: False
2026-02-03 10:54:53,656 - root - INFO -   Using LoRA: True
2026-02-03 10:54:53,656 - root - INFO -
================================================================================
2026-02-03 10:54:53,656 - root - INFO - STEP 3: TRAINING MODEL
2026-02-03 10:54:53,657 - root - INFO - ================================================================================

2026-02-03 10:54:53,657 - root - INFO - ================================================================================
2026-02-03 10:54:53,657 - root - INFO - Starting MLM Training
2026-02-03 10:54:53,657 - root - INFO - ================================================================================
2026-02-03 10:54:53,657 - root - INFO - Training configuration:
2026-02-03 10:54:53,657 - root - INFO -   Epochs: 10
2026-02-03 10:54:53,657 - root - INFO -   Batch size: 8
2026-02-03 10:54:53,657 - root - INFO -   Learning rate: 0.0002
2026-02-03 10:54:53,657 - root - INFO -   MLM probability: 0.15
2026-02-03 10:54:53,657 - root - INFO -   Gradient accumulation: 4
2026-02-03 10:54:53,658 - root - INFO -   Effective batch size: 32
2026-02-03 10:54:53,658 - root - INFO -   FP16: False
2026-02-03 10:54:53,883 - root - INFO - Starting training...
{'loss': 13.4417, 'grad_norm': 6.842384338378906, 'learning_rate': 3.960000000000001e-05, 'epoch': 0.9}
{'loss': 8.6711, 'grad_norm': 4.100255012512207, 'learning_rate': 7.960000000000001e-05, 'epoch': 1.79}
{'loss': 7.2096, 'grad_norm': 4.633697509765625, 'learning_rate': 0.00011960000000000001, 'epoch': 2.68}
{'loss': 6.417, 'grad_norm': 4.373517036437988, 'learning_rate': 0.0001596, 'epoch': 3.58}
{'loss': 5.8438, 'grad_norm': 4.859941005706787, 'learning_rate': 0.0001996, 'epoch': 4.47}
{'eval_loss': 5.324168682098389, 'eval_runtime': 175.9932, 'eval_samples_per_second': 2.25, 'eval_steps_per_second': 0.284, 'epoch': 4.47}
{'loss': 5.4605, 'grad_norm': 4.412198066711426, 'learning_rate': 0.00016806451612903228, 'epoch': 5.36}
{'loss': 5.2044, 'grad_norm': 4.788223743438721, 'learning_rate': 0.00013580645161290325, 'epoch': 6.25}
{'loss': 4.9992, 'grad_norm': 4.442352771759033, 'learning_rate': 0.0001035483870967742, 'epoch': 7.14}
{'loss': 4.8583, 'grad_norm': 4.487061500549316, 'learning_rate': 7.129032258064517e-05, 'epoch': 8.04}
{'loss': 4.7813, 'grad_norm': 3.9753506183624268, 'learning_rate': 3.903225806451613e-05, 'epoch': 8.93}
{'eval_loss': 4.509352207183838, 'eval_runtime': 208.7924, 'eval_samples_per_second': 1.897, 'eval_steps_per_second': 0.239, 'epoch': 8.93}
{'loss': 4.7277, 'grad_norm': 3.9955785274505615, 'learning_rate': 6.774193548387098e-06, 'epoch': 9.83}
{'train_runtime': 53358.2637, 'train_samples_per_second': 0.666, 'train_steps_per_second': 0.021, 'train_loss': 6.478477948052543, 'epoch': 10.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1120/1120 [14:49:18<00:00, 47.64s/it]
2026-02-04 01:44:12,401 - root - INFO - Training completed!
2026-02-04 01:44:12,401 - root - INFO - Training loss: 6.4785
2026-02-04 01:44:12,401 - root - INFO - Evaluating model...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:34<00:00,  3.10s/it]
2026-02-04 01:47:26,221 - root - INFO - Evaluation results:
2026-02-04 01:47:26,222 - root - INFO -   eval_loss: 4.4588
2026-02-04 01:47:26,222 - root - INFO -   eval_runtime: 193.8150
2026-02-04 01:47:26,222 - root - INFO -   eval_samples_per_second: 2.0430
2026-02-04 01:47:26,222 - root - INFO -   eval_steps_per_second: 0.2580
2026-02-04 01:47:26,222 - root - INFO -   epoch: 10.0000
2026-02-04 01:47:26,222 - root - INFO - Total training time: 14:52:32.565238
2026-02-04 01:47:26,222 - root - INFO - ================================================================================
2026-02-04 01:47:26,222 - root - INFO - ✓ Training complete
--- Logging error ---
Traceback (most recent call last):
  File "C:\Python313\Lib\logging\__init__.py", line 1153, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 40: character maps to <undefined>
Call stack:
  File "C:\projects\mlm-training\example_mlm_training.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\example_mlm_training.py", line 166, in main
    logging.info(f"✓ Training complete")
Message: '✓ Training complete'
Arguments: ()

(mlm-training) C:\projects\mlm-training>

