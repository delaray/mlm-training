
(mlm-training) C:\projects\mlm-training>python mlm_train.py
2026-02-05 10:09:41,437 - root - INFO - ================================================================================
2026-02-05 10:09:41,437 - root - INFO - MLM TRAINING PIPELINE
2026-02-05 10:09:41,437 - root - INFO - ================================================================================
2026-02-05 10:09:41,437 - root - INFO - Model: microsoft/deberta-v3-base
2026-02-05 10:09:41,438 - root - INFO - Data Directory: data/books
2026-02-05 10:09:41,438 - root - INFO - Output Directory: results\training-deberta-v3-base-20260205-1009
2026-02-05 10:09:41,438 - root - INFO - ================================================================================
2026-02-05 10:09:41,439 - root - INFO -
================================================================================
2026-02-05 10:09:41,439 - root - INFO - STEP 1: PREPARING DATASET
2026-02-05 10:09:41,439 - root - INFO - ================================================================================

2026-02-05 10:09:41,439 - root - INFO - ================================================================================
2026-02-05 10:09:41,439 - root - INFO - Starting MLM Dataset Preparation
2026-02-05 10:09:41,439 - root - INFO - ================================================================================
2026-02-05 10:09:41,439 - root - INFO - Loading tokenizer from: models\deberta-v3-base
2026-02-05 10:09:41,647 - root - INFO - Reading documents from: data/books
Document loader: <src.ingest.DocumentIngest object at 0x000001B7C51A01A0>
Reading files in data/books
MuPDF error: syntax error: cannot find ExtGState resource 'pgf@CA1'

MuPDF error: syntax error: cannot find ExtGState resource 'pgf@ca1'

MuPDF error: syntax error: cannot find ExtGState resource 'pgf@CA1'

MuPDF error: syntax error: cannot find ExtGState resource 'pgf@ca1'

2026-02-05 10:09:59,535 - root - INFO - Successfully read 38 files
2026-02-05 10:09:59,535 - root - INFO - Total chunks available: 13876
2026-02-05 10:09:59,690 - root - INFO - Splitting dataset (test_split=0.1)
2026-02-05 10:09:59,693 - root - INFO - Tokenizing dataset...
Tokenizing (num_proc=4): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12488/12488 [00:11<00:00, 1095.10 examples/s]
Tokenizing (num_proc=4): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1388/1388 [00:09<00:00, 145.24 examples/s]
2026-02-05 10:10:20,694 - root - INFO - Training samples: 12488
2026-02-05 10:10:20,694 - root - INFO - Test samples: 1388
2026-02-05 10:10:20,694 - root - INFO - Max sequence length: 512
2026-02-05 10:10:20,694 - root - INFO - Dataset preparation completed in 0:00:39.254795
2026-02-05 10:10:20,694 - root - INFO - ================================================================================
2026-02-05 10:10:20,707 - root - INFO - ✓ Dataset prepared successfully
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
2026-02-05 10:10:20,734 - root - INFO -   Training samples: 12488
2026-02-05 10:10:20,734 - root - INFO -   Test samples: 1388
2026-02-05 10:10:20,734 - root - INFO -
================================================================================
2026-02-05 10:10:20,734 - root - INFO - STEP 2: SETTING UP MODEL
2026-02-05 10:10:20,734 - root - INFO - ================================================================================

2026-02-05 10:10:20,734 - root - INFO - ================================================================================
2026-02-05 10:10:20,734 - root - INFO - Setting up model for MLM training
2026-02-05 10:10:20,734 - root - INFO - ================================================================================
2026-02-05 10:10:20,734 - root - INFO - Loading model without quantization (on CPU first)
`torch_dtype` is deprecated! Use `dtype` instead!
2026-02-05 10:10:20,794 - root - INFO - Model loaded on CPU
2026-02-05 10:10:20,794 - root - INFO - Applying LoRA configuration on CPU
2026-02-05 10:10:20,794 - root - INFO - Auto-detected target modules: ['query', 'key', 'value', 'dense']
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
W0205 10:10:21.106000 23760 .venv\Lib\site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
trainable params: 1,794,048 || all params: 186,345,828 || trainable%: 0.9628
2026-02-05 10:10:21,780 - root - INFO - LoRA applied successfully on CPU
2026-02-05 10:10:21,780 - root - INFO - Keeping model on CPU (RTX 5090 sm_120 lacks required CUDA kernels)
2026-02-05 10:10:21,780 - root - INFO - Note: Training will use CPU. For GPU training, wait for PyTorch with full sm_120 support.
2026-02-05 10:10:21,780 - root - INFO - ================================================================================
2026-02-05 10:10:21,780 - root - INFO - ✓ Model setup complete
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
2026-02-05 10:10:21,781 - root - INFO -   Quantized: False
2026-02-05 10:10:21,781 - root - INFO -   Using LoRA: True
2026-02-05 10:10:21,782 - root - INFO -
================================================================================
2026-02-05 10:10:21,782 - root - INFO - STEP 3: TRAINING MODEL
2026-02-05 10:10:21,782 - root - INFO - ================================================================================

2026-02-05 10:10:21,782 - root - INFO - ================================================================================
2026-02-05 10:10:21,782 - root - INFO - Starting MLM Training
2026-02-05 10:10:21,782 - root - INFO - ================================================================================
2026-02-05 10:10:21,783 - root - INFO - Training configuration:
2026-02-05 10:10:21,783 - root - INFO -   Epochs: 10
2026-02-05 10:10:21,783 - root - INFO -   Batch size: 8
2026-02-05 10:10:21,783 - root - INFO -   Learning rate: 0.0002
2026-02-05 10:10:21,783 - root - INFO -   MLM probability: 0.15
2026-02-05 10:10:21,783 - root - INFO -   Gradient accumulation: 4
2026-02-05 10:10:21,783 - root - INFO -   Effective batch size: 32
2026-02-05 10:10:21,783 - root - INFO -   FP16: False
2026-02-05 10:10:21,943 - root - INFO - Starting training...
{'loss': 13.4577, 'grad_norm': 6.778571605682373, 'learning_rate': 3.960000000000001e-05, 'epoch': 0.26}
{'loss': 8.3576, 'grad_norm': 4.411673545837402, 'learning_rate': 7.960000000000001e-05, 'epoch': 0.51}
{'loss': 6.7686, 'grad_norm': 4.872111797332764, 'learning_rate': 0.00011960000000000001, 'epoch': 0.77}
{'loss': 6.0498, 'grad_norm': 5.336423397064209, 'learning_rate': 0.0001596, 'epoch': 1.02}
{'loss': 5.5959, 'grad_norm': 5.361867904663086, 'learning_rate': 0.0001996, 'epoch': 1.28}
{'eval_loss': 5.208728790283203, 'eval_runtime': 505.6632, 'eval_samples_per_second': 2.745, 'eval_steps_per_second': 0.344, 'epoch': 1.28}
{'loss': 5.2848, 'grad_norm': 4.723018646240234, 'learning_rate': 0.00019419354838709678, 'epoch': 1.54}
{'loss': 5.0387, 'grad_norm': 4.823625087738037, 'learning_rate': 0.0001883284457478006, 'epoch': 1.79}
{'loss': 4.8324, 'grad_norm': 4.070242881774902, 'learning_rate': 0.00018246334310850442, 'epoch': 2.05}
{'loss': 4.6347, 'grad_norm': 4.009469509124756, 'learning_rate': 0.00017659824046920822, 'epoch': 2.3}
{'loss': 4.4686, 'grad_norm': 4.284495830535889, 'learning_rate': 0.00017073313782991201, 'epoch': 2.56}
{'eval_loss': 4.134799003601074, 'eval_runtime': 592.1033, 'eval_samples_per_second': 2.344, 'eval_steps_per_second': 0.294, 'epoch': 2.56}
{'loss': 4.3232, 'grad_norm': 4.578067302703857, 'learning_rate': 0.00016486803519061584, 'epoch': 2.81}
{'loss': 4.2339, 'grad_norm': 4.075802803039551, 'learning_rate': 0.00015900293255131966, 'epoch': 3.07}
{'loss': 4.1159, 'grad_norm': 3.785128355026245, 'learning_rate': 0.00015313782991202348, 'epoch': 3.33}
{'loss': 4.0419, 'grad_norm': 3.698971748352051, 'learning_rate': 0.00014727272727272728, 'epoch': 3.58}
{'loss': 3.9695, 'grad_norm': 3.4979121685028076, 'learning_rate': 0.0001414076246334311, 'epoch': 3.84}
{'eval_loss': 3.642103910446167, 'eval_runtime': 525.9367, 'eval_samples_per_second': 2.639, 'eval_steps_per_second': 0.331, 'epoch': 3.84}
{'loss': 3.8864, 'grad_norm': 3.5144033432006836, 'learning_rate': 0.0001355425219941349, 'epoch': 4.09}
{'loss': 3.8379, 'grad_norm': 3.6484696865081787, 'learning_rate': 0.00012967741935483872, 'epoch': 4.35}
{'loss': 3.7916, 'grad_norm': 3.519408702850342, 'learning_rate': 0.00012381231671554252, 'epoch': 4.6}
{'loss': 3.723, 'grad_norm': 3.167036533355713, 'learning_rate': 0.00011794721407624634, 'epoch': 4.86}
{'loss': 3.6985, 'grad_norm': 3.3662874698638916, 'learning_rate': 0.00011208211143695015, 'epoch': 5.12}
{'eval_loss': 3.3912153244018555, 'eval_runtime': 510.9126, 'eval_samples_per_second': 2.717, 'eval_steps_per_second': 0.341, 'epoch': 5.12}
{'loss': 3.6631, 'grad_norm': 3.4093291759490967, 'learning_rate': 0.00010621700879765397, 'epoch': 5.37}
{'loss': 3.6041, 'grad_norm': 3.486290454864502, 'learning_rate': 0.00010035190615835777, 'epoch': 5.63}
{'loss': 3.5842, 'grad_norm': 3.41633939743042, 'learning_rate': 9.448680351906158e-05, 'epoch': 5.88}
{'loss': 3.5596, 'grad_norm': 3.5887300968170166, 'learning_rate': 8.86217008797654e-05, 'epoch': 6.14}
{'loss': 3.5055, 'grad_norm': 3.66412091255188, 'learning_rate': 8.275659824046921e-05, 'epoch': 6.39}
{'eval_loss': 3.251267194747925, 'eval_runtime': 622.6159, 'eval_samples_per_second': 2.229, 'eval_steps_per_second': 0.279, 'epoch': 6.39}
{'loss': 3.492, 'grad_norm': 3.2770373821258545, 'learning_rate': 7.689149560117302e-05, 'epoch': 6.65}
{'loss': 3.494, 'grad_norm': 3.350437879562378, 'learning_rate': 7.102639296187683e-05, 'epoch': 6.91}
{'loss': 3.4568, 'grad_norm': 3.1917059421539307, 'learning_rate': 6.516129032258065e-05, 'epoch': 7.16}
{'loss': 3.4164, 'grad_norm': 3.184708595275879, 'learning_rate': 5.9296187683284464e-05, 'epoch': 7.42}
{'loss': 3.4183, 'grad_norm': 3.3286798000335693, 'learning_rate': 5.3431085043988274e-05, 'epoch': 7.67}
{'eval_loss': 3.1670970916748047, 'eval_runtime': 529.4131, 'eval_samples_per_second': 2.622, 'eval_steps_per_second': 0.329, 'epoch': 7.67}
{'loss': 3.4379, 'grad_norm': 3.2747459411621094, 'learning_rate': 4.7565982404692084e-05, 'epoch': 7.93}
{'loss': 3.426, 'grad_norm': 3.45760178565979, 'learning_rate': 4.170087976539589e-05, 'epoch': 8.18}
{'loss': 3.4129, 'grad_norm': 3.2671430110931396, 'learning_rate': 3.583577712609971e-05, 'epoch': 8.44}
{'loss': 3.3638, 'grad_norm': 3.2969956398010254, 'learning_rate': 2.997067448680352e-05, 'epoch': 8.7}
{'loss': 3.3652, 'grad_norm': 3.0147063732147217, 'learning_rate': 2.4105571847507332e-05, 'epoch': 8.95}
{'eval_loss': 3.0774221420288086, 'eval_runtime': 535.8653, 'eval_samples_per_second': 2.59, 'eval_steps_per_second': 0.325, 'epoch': 8.95}
{'loss': 3.3577, 'grad_norm': 3.3737146854400635, 'learning_rate': 1.8240469208211145e-05, 'epoch': 9.21}
{'loss': 3.3653, 'grad_norm': 3.6020736694335938, 'learning_rate': 1.2375366568914957e-05, 'epoch': 9.46}
{'loss': 3.3285, 'grad_norm': 3.2337372303009033, 'learning_rate': 6.5102639296187695e-06, 'epoch': 9.72}
{'loss': 3.3463, 'grad_norm': 3.2016236782073975, 'learning_rate': 6.451612903225807e-07, 'epoch': 9.98}
{'train_runtime': 177027.6262, 'train_samples_per_second': 0.705, 'train_steps_per_second': 0.022, 'train_loss': 4.349117130757597, 'epoch': 10.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3910/3910 [49:10:27<00:00, 45.28s/it]
2026-02-07 11:20:49,830 - root - INFO - Training completed!
2026-02-07 11:20:49,830 - root - INFO - Training loss: 4.3491
2026-02-07 11:20:49,831 - root - INFO - Evaluating model...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 174/174 [08:07<00:00,  2.80s/it]
2026-02-07 11:29:17,249 - root - INFO - Evaluation results:
2026-02-07 11:29:17,249 - root - INFO -   eval_loss: 3.1398
2026-02-07 11:29:17,249 - root - INFO -   eval_runtime: 507.4124
2026-02-07 11:29:17,249 - root - INFO -   eval_samples_per_second: 2.7350
2026-02-07 11:29:17,250 - root - INFO -   eval_steps_per_second: 0.3430
2026-02-07 11:29:17,250 - root - INFO -   epoch: 10.0000
2026-02-07 11:29:17,250 - root - INFO - Total training time: 2 days, 1:18:55.467753
2026-02-07 11:29:17,250 - root - INFO - ================================================================================
2026-02-07 11:29:17,250 - root - INFO - ✓ Training complete
--- Logging error ---
Traceback (most recent call last):
  File "C:\Python313\Lib\logging\__init__.py", line 1153, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python313\Lib\encodings\
