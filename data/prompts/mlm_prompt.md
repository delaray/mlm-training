# Project Overview
The purpose of this project is multifold:
1. Further train an existing encoder model using MLM training
2. Prepare a synthetic SFT dataset by asking questions on existing book paragraphs
3. Fint-tune an existing open-source decoder model using the previously prepared synthetic data set

# Hardware context
I have a Windows 11 PC with a 24G GPU and about 200G of RAM. 

# Software context 
Look through the code in the src sub-directory of the project directory so you understand the code and installed libraries I already have. I am using uv as my project and dependency management. I have a virtual environment already set up called venv. All code should be run in that environment.

# Task Overview
The pupose of this task is to further train an existing encoder model like newBert (or a smaller model given my hardware) using MLM training. Feel free to recommend and select an open-source model. All models should be stored in the models subdirectory which is ignored by git.

# Task Details
Please perform the following consecutive tasks:
1. Write a function that prepares a dataset from a directory of pdf books for MLM training of an open-soource encoder model like newBert (or a smaller model given my hardware). The dataset should be suitable for use with the HF transformers library trainer class
2. Recommend a couple of open-source models that I can download from HF and provide links to their model card. Provide instructions for downloading the model
3. Write a function that takes the dataset created in (1) and an encoder model from (2) as arguments and further trains the model using MLM. Training time is not an issue. Any optimizations should be for model size and quality, i.e. PEFT techniques like LORA, QLORA, quantization, etc...
4. Provide functions to save and load the newly trained model
5. Write a function that takes the model name and text and generates an embedding vector of the text. Provide relevant hyperparameters like embedding size etc..
6. Finally provide a short summary of the work performed and any instructions or relevant details that I need to run the code.