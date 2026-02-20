(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Deep-Learning-Book--MIT-Press--2016.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\deep-learning-book-sft.jsonl
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 788/788 [1:24:09<00:00,  6.41s/it]

============================================================
✅ Done! Generated 2398 chat examples
📁 Output: results\deep-learning-book-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Understanding-Deep-Learning--1st--MIT-Press--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\understanding-deep-learning-sft.jsonl
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 760/760 [1:16:53<00:00,  6.07s/it]

============================================================
✅ Done! Generated 2160 chat examples
📁 Output: results\understanding-deep-learning-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Deep-Learning-with-Python--3rd--Manning--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\deep-learning-with-python-sft.jsonl
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 825/825 [1:21:23<00:00,  5.92s/it]

============================================================
✅ Done! Generated 2427 chat examples
⚠️  Skipped 2 paragraphs due to repeated errors
📁 Output: results\deep-learning-with-python-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Natural-Language-Processing-in-Action--2nd--Manning--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\nlp-in-action-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1051/1051 [1:47:42<00:00,  6.15s/it]

============================================================
✅ Done! Generated 2941 chat examples
⚠️  Skipped 4 paragraphs due to repeated errors
📁 Output: results\nlp-in-action-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Generative-AI-in-Action--Final--Manning--2024.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\genai-in-action-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 592/592 [59:11<00:00,  6.00s/it]

============================================================
✅ Done! Generated 1860 chat examples
⚠️  Skipped 2 paragraphs due to repeated errors
📁 Output: results\genai-in-action-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\The-RLHF-Book--v2--Manning--2026.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\rlhf-book-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 299/299 [31:42<00:00,  6.36s/it]

============================================================
✅ Done! Generated 860 chat examples
📁 Output: results\rlhf-book-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Reinforcement-Learning-An-Introduction--2nd--MIT-Press--2017.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\rl-introduction-sft.jsonl
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 893/893 [1:32:44<00:00,  6.23s/it]

============================================================
✅ Done! Generated 2760 chat examples
📁 Output: results\rl-introduction-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Graph-Neural-Networks-in-Action--Manning--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\gnn-in-action-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 457/457 [45:54<00:00,  6.03s/it]

============================================================
✅ Done! Generated 1390 chat examples
📁 Output: results\gnn-in-action-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\The-Principles-of-Diffusion-Models--Stanford-University--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\principles-of-diffusion-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 432/432 [47:22<00:00,  6.58s/it]

============================================================
✅ Done! Generated 1377 chat examples
📁 Output: results\principles-of-diffusion-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Machine-Learning-Techniques-in-Image-Processing-and-Computer-Vision--1st--CRC-Press--2024.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\ml-techniques-for-cv-sft.jsonl
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 329/329 [32:33<00:00,  5.94s/it]

============================================================
✅ Done! Generated 842 chat examples
📁 Output: results\ml-techniques-for-cv-sft.jsonl
============================================================


(mlm-training) C:\projects\mlm-training>python src\sft_data.py data\books\Knowledge-Graphs-and-LLMs-in-Action--Manning--2025.pdf --model llama3.1:8b --pairs-per-paragraph 4 --out results\gnn-and-llm-in-action-sft.jsonl
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 698/698 [1:12:32<00:00,  6.24s/it]

============================================================
✅ Done! Generated 2096 chat examples
⚠️  Skipped 1 paragraphs due to repeated errors
📁 Output: results\gnn-and-llm-in-action-sft.jsonl
============================================================
