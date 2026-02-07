(mlm-training) C:\projects\mlm-training>python mlm_train.py

2026-02-04 11:24:35,582 - root - INFO - ================================================================================
2026-02-04 11:24:35,582 - root - INFO - MLM TRAINING PIPELINE
2026-02-04 11:24:35,583 - root - INFO - ================================================================================
2026-02-04 11:24:35,583 - root - INFO - Model: microsoft/deberta-v3-base
2026-02-04 11:24:35,583 - root - INFO - Data Directory: data/books
2026-02-04 11:24:35,583 - root - INFO - Output Directory: results\training-deberta-v3-base-20260204-1124
2026-02-04 11:24:35,583 - root - INFO - ================================================================================

2026-02-04 11:24:35,585 - root - INFO -
================================================================================
2026-02-04 11:24:35,585 - root - INFO - STEP 1: PREPARING DATASET
2026-02-04 11:24:35,585 - root - INFO - ================================================================================

2026-02-04 11:24:35,585 - root - INFO - ================================================================================
2026-02-04 11:24:35,585 - root - INFO - Starting MLM Dataset Preparation
2026-02-04 11:24:35,585 - root - INFO - ================================================================================
2026-02-04 11:24:35,585 - root - INFO - Loading tokenizer from: models\deberta-v3-base
2026-02-04 11:24:35,814 - root - INFO - Reading documents from: data/books
Document loader: <src.ingest.DocumentIngest object at 0x000001C69AC601A0>
Reading files in data/books
2026-02-04 11:24:41,832 - root - INFO - Successfully read 22 files
2026-02-04 11:24:41,832 - root - INFO - Total chunks available: 5394
2026-02-04 11:24:41,891 - root - INFO - Splitting dataset (test_split=0.1)
2026-02-04 11:24:41,894 - root - INFO - Tokenizing dataset...
Tokenizing (num_proc=4): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4854/4854 [00:09<00:00, 531.09 examples/s]
Tokenizing (num_proc=4): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 540/540 [00:08<00:00, 66.36 examples/s]
2026-02-04 11:24:59,209 - root - INFO - Training samples: 4854
2026-02-04 11:24:59,209 - root - INFO - Test samples: 540
2026-02-04 11:24:59,209 - root - INFO - Max sequence length: 512
2026-02-04 11:24:59,210 - root - INFO - Dataset preparation completed in 0:00:23.624430
2026-02-04 11:24:59,210 - root - INFO - ================================================================================
2026-02-04 11:24:59,215 - root - INFO - ✓ Dataset prepared successfully
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
  File "C:\projects\mlm-training\mlm_train.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\mlm_train.py", line 118, in main
    logging.info(f"✓ Dataset prepared successfully")
Message: '✓ Dataset prepared successfully'
Arguments: ()
2026-02-04 11:24:59,240 - root - INFO -   Training samples: 4854
2026-02-04 11:24:59,240 - root - INFO -   Test samples: 540

2026-02-04 11:24:59,240 - root - INFO -
================================================================================
2026-02-04 11:24:59,240 - root - INFO - STEP 2: SETTING UP MODEL
2026-02-04 11:24:59,240 - root - INFO - ================================================================================

2026-02-04 11:24:59,241 - root - INFO - ================================================================================
2026-02-04 11:24:59,241 - root - INFO - Setting up model for MLM training
2026-02-04 11:24:59,241 - root - INFO - ================================================================================
2026-02-04 11:24:59,241 - root - INFO - Loading model without quantization (on CPU first)
`torch_dtype` is deprecated! Use `dtype` instead!
2026-02-04 11:24:59,311 - root - INFO - Model loaded on CPU
2026-02-04 11:24:59,312 - root - INFO - Applying LoRA configuration on CPU
2026-02-04 11:24:59,312 - root - INFO - Auto-detected target modules: ['query', 'key', 'value', 'dense']
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
W0204 11:24:59.607000 21552 .venv\Lib\site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
trainable params: 1,794,048 || all params: 186,345,828 || trainable%: 0.9628
2026-02-04 11:25:00,363 - root - INFO - LoRA applied successfully on CPU
2026-02-04 11:25:00,363 - root - INFO - Keeping model on CPU (RTX 5090 sm_120 lacks required CUDA kernels)
2026-02-04 11:25:00,363 - root - INFO - Note: Training will use CPU. For GPU training, wait for PyTorch with full sm_120 support.
2026-02-04 11:25:00,364 - root - INFO - ================================================================================
2026-02-04 11:25:00,364 - root - INFO - ✓ Model setup complete
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
  File "C:\projects\mlm-training\mlm_train.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\mlm_train.py", line 141, in main
    logging.info(f"✓ Model setup complete")
Message: '✓ Model setup complete'
Arguments: ()
2026-02-04 11:25:00,365 - root - INFO -   Quantized: False
2026-02-04 11:25:00,365 - root - INFO -   Using LoRA: True

2026-02-04 11:25:00,365 - root - INFO -
================================================================================
2026-02-04 11:25:00,365 - root - INFO - STEP 3: TRAINING MODEL
2026-02-04 11:25:00,365 - root - INFO - ================================================================================

2026-02-04 11:25:00,365 - root - INFO - ================================================================================
2026-02-04 11:25:00,365 - root - INFO - Starting MLM Training
2026-02-04 11:25:00,365 - root - INFO - ================================================================================
2026-02-04 11:25:00,366 - root - INFO - Training configuration:
2026-02-04 11:25:00,366 - root - INFO -   Epochs: 10
2026-02-04 11:25:00,366 - root - INFO -   Batch size: 8
2026-02-04 11:25:00,366 - root - INFO -   Learning rate: 0.0002
2026-02-04 11:25:00,366 - root - INFO -   MLM probability: 0.15
2026-02-04 11:25:00,366 - root - INFO -   Gradient accumulation: 4
2026-02-04 11:25:00,367 - root - INFO -   Effective batch size: 32
2026-02-04 11:25:00,367 - root - INFO -   FP16: False
2026-02-04 11:25:00,569 - root - INFO - Starting training...
{'loss': 13.4089, 'grad_norm': 6.834702491760254, 'learning_rate': 3.960000000000001e-05, 'epoch': 0.66}
{'loss': 8.7287, 'grad_norm': 3.7258353233337402, 'learning_rate': 7.960000000000001e-05, 'epoch': 1.32}
{'loss': 7.27, 'grad_norm': 5.002772331237793, 'learning_rate': 0.00011960000000000001, 'epoch': 1.98}
{'loss': 6.5075, 'grad_norm': 5.1708197593688965, 'learning_rate': 0.0001596, 'epoch': 2.63}
{'loss': 6.077, 'grad_norm': 5.63308048248291, 'learning_rate': 0.0001996, 'epoch': 3.29}
{'eval_loss': 5.653354167938232, 'eval_runtime': 202.6165, 'eval_samples_per_second': 2.665, 'eval_steps_per_second': 0.336, 'epoch': 3.29}
{'loss': 5.7321, 'grad_norm': 4.549376010894775, 'learning_rate': 0.00018058823529411766, 'epoch': 3.95}
{'loss': 5.5041, 'grad_norm': 4.502525806427002, 'learning_rate': 0.00016098039215686275, 'epoch': 4.61}
{'loss': 5.2999, 'grad_norm': 4.399777889251709, 'learning_rate': 0.00014137254901960787, 'epoch': 5.26}
{'loss': 5.158, 'grad_norm': 3.8689756393432617, 'learning_rate': 0.00012176470588235293, 'epoch': 5.92}
{'loss': 4.9977, 'grad_norm': 4.214856147766113, 'learning_rate': 0.00010215686274509803, 'epoch': 6.58}
{'eval_loss': 4.715693950653076, 'eval_runtime': 262.8258, 'eval_samples_per_second': 2.055, 'eval_steps_per_second': 0.259, 'epoch': 6.58}
{'loss': 4.9129, 'grad_norm': 4.605369567871094, 'learning_rate': 8.254901960784314e-05, 'epoch': 7.24}
{'loss': 4.8443, 'grad_norm': 3.8419158458709717, 'learning_rate': 6.294117647058824e-05, 'epoch': 7.9}
{'loss': 4.7882, 'grad_norm': 4.01943302154541, 'learning_rate': 4.3333333333333334e-05, 'epoch': 8.55}
{'loss': 4.7178, 'grad_norm': 4.21505069732666, 'learning_rate': 2.372549019607843e-05, 'epoch': 9.21}
{'loss': 4.7332, 'grad_norm': 3.770056962966919, 'learning_rate': 4.11764705882353e-06, 'epoch': 9.87}
{'eval_loss': 4.470961570739746, 'eval_runtime': 261.4162, 'eval_samples_per_second': 2.066, 'eval_steps_per_second': 0.26, 'epoch': 9.87}
{'train_runtime': 69871.535, 'train_samples_per_second': 0.695, 'train_steps_per_second': 0.022, 'train_loss': 6.1592762595728825, 'epoch': 10.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1520/1520 [19:24:31<00:00, 45.97s/it]
2026-02-05 06:49:32,374 - root - INFO - Training completed!
2026-02-05 06:49:32,375 - root - INFO - Training loss: 6.1593
2026-02-05 06:49:32,375 - root - INFO - Evaluating model...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 68/68 [03:40<00:00,  3.25s/it]
2026-02-05 06:53:34,736 - root - INFO - Evaluation results:
2026-02-05 06:53:34,736 - root - INFO -   eval_loss: 4.7352
2026-02-05 06:53:34,736 - root - INFO -   eval_runtime: 242.3551
2026-02-05 06:53:34,736 - root - INFO -   eval_samples_per_second: 2.2280
2026-02-05 06:53:34,736 - root - INFO -   eval_steps_per_second: 0.2810
2026-02-05 06:53:34,736 - root - INFO -   epoch: 10.0000
2026-02-05 06:53:34,736 - root - INFO - Total training time: 19:28:34.371072
2026-02-05 06:53:34,736 - root - INFO - ================================================================================
2026-02-05 06:53:34,737 - root - INFO - ✓ Training complete
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
  File "C:\projects\mlm-training\mlm_train.py", line 326, in <module>
    main()
  File "C:\projects\mlm-training\mlm_train.py", line 166, in main
    logging.info(f"✓ Training complete")
Message: '✓ Training complete'
Arguments: ()

(mlm-training) C:\projects\mlm-training>
