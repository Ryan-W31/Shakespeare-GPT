# Shakespeare-GPT
Building a Generative Pre-Trained Transformer (GPT) to generate Shakespeare-like text from scratch!

This repo is inspired by and follows along Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) youtube playlist.

In this repo, I create a GPT trained on the [tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) which contains over 1 million characters.

The GPT implemented in [shakespeare-gpt.py](./shakespeare-gpt.py) is a character-level Bigram GPT model. The final settings of this model resulted in more than 10 million trainable parameters, achieving the results found in [output.txt](./output.txt)

---

#### Disclaimer
The Bigram GPT Model was trained by using [Google Colab](https://colab.google/)'s T4 GPU with CUDA capabilities. Even so, the model took ~45 minutes to train. If you are trying to clone or replicate this repo, I suggest using Google Colab T4 GPU or your own GPU (specifically NVIDIA GPUs) to train the model.

---

To test this repo yourself, follow these steps:
1. Use the command ```git clone https://github.com/Ryan-W31/Shakespeare-GPT.git```

- If running locally
    1. It is recommended to create a virtual environment, after doing so, all packages can be found in requirements.txt.
       ```pip install -r requirements.txt```
    2. Run the python script.
       ```python shakespeare-gpt.py```
- If using Google Colab:
    1. Create a new notebook.
    2. Upload input.txt and shakespeare-gpt.py to your Google Drive.
    3. Within the notebook, mount your Google Drive.
    4. Navigate to where input.txt and shakespeare-gpt.py are in your Google Drive using ```%cd```.
    5. Ensure you are using a GPU or TPU runtime, then run: ```!python shakespeare-gpt.py```        