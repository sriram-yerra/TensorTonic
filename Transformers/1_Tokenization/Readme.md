### 1. **What tokenization does in Transformers**
- **Tokenization** converts raw text into **tokens (subwords/words/characters)** and then maps them to **integer IDs**.
- This is necessary because **transformers operate on numbers (tensors), not raw text**. :contentReference[oaicite:0]{index=0}  

**Flow:**
Text → Tokens → Token IDs → Embeddings → Transformer

---

### 2. **Typical tokenization output format**
When you tokenize a sentence using a transformer tokenizer (like BERT), the output is:

- **input_ids** → numerical IDs of tokens  
- **token_type_ids** → segment identifiers (used in tasks like sentence pairs)  
- **attention_mask** → tells model which tokens to attend to  

Example:
"hello world" → [CLS] hello world [SEP]  
input_ids = [101, 19082, 1362, 102]

These IDs represent tokens in the vocabulary. :contentReference[oaicite:1]{index=1}  

---

### 3. **Why we cannot directly pass token lists**
- Tokenizer output is usually a **Python list**
- Transformers require **tensors (PyTorch/TF)** as input

So we must convert:  
list → tensor

before feeding into the model.

---

### 4. **Types of tokenization methods**
Common methods used in transformers:

- **Character-level**
- **Word-level**
- **Subword-level (most common)**  
  - BPE  
  - WordPiece  
  - SentencePiece  

These help handle **unknown words and vocabulary size efficiently**. :contentReference[oaicite:2]{index=2}  

---

### 5. **Padding and truncation (important in the question)**
Since transformers expect **fixed-length sequences**:

- **Padding** → add `[PAD]` tokens to shorter sequences  
- **Truncation** → cut long sequences  

This ensures all sequences are same length and efficient for batch processing. :contentReference[oaicite:3]{index=3}

---

## Final Short Answer (what you should write)

**Tokenization in transformers converts text into tokens, maps them to integer IDs, adds special tokens like `[CLS]` and `[SEP]`, and outputs structures like `input_ids`, `token_type_ids`, and `attention_mask`, which are then converted into tensors and passed into the model after padding/truncation.**

---

## SimpleTokenizer — Complete Solution

Below is the **correct implementation** for the TensorTonic tokenization problem.

---

### 1. **Key Logic**
- Add **special tokens first** with fixed IDs  
`<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`
- Then assign IDs to **unique words from training texts**
- Encoding maps words → IDs
- Decoding maps IDs → words