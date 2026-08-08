# Chapter 5 Bonus Technologies Explained

This guide explains the bonus materials and alternative LLM architectures covered in subfolders `02` to `18` of Chapter 5. These topics cover advanced training techniques, alternative models, speed optimizations, and full from-scratch implementations of modern state-of-the-art architectures.

---

## 02. Alternative Weight Loading (`02_alternative_weight_loading`)
**What is it?** Provides code to load the GPT model weights from alternative sources in case they are unavailable directly from OpenAI.
**Technology:** Advanced PyTorch checkpoint loading and dictionary manipulation.

## 03. Pretraining on Project Gutenberg (`03_bonus_pretraining_on_gutenberg`)
**What is it?** A guide and code to pretrain the LLM on a larger scale using the entire corpus of public domain books from Project Gutenberg.
**Technology:** Unsupervised pretraining at scale and efficient text data ingestion.

## 04. Learning Rate Schedulers (`04_learning_rate_schedulers`)
**What is it?** Implements a more sophisticated training loop using advanced optimization techniques.
**Technology:** **Learning Rate Schedulers** (dynamically adjusting the learning rate during training for better convergence) and **Gradient Clipping** (preventing exploding gradients by capping their maximum value).

## 05. Hyperparameter Tuning (`05_bonus_hparam_tuning`)
**What is it?** An optional script showing how to systematically search for the best model configuration.
**Technology:** Hyperparameter Optimization (HPO) for deep learning.

## 06. User Interface (`06_user_interface`)
**What is it?** Implements an interactive graphical user interface (GUI) to chat with your pretrained LLM.
**Technology:** Python UI and web frameworks.

## 07. GPT to LLaMA 3.2 (`07_gpt_to_llama`)
**What is it?** A step-by-step guide converting the foundational GPT architecture from the book into the modern Llama 3.2 architecture, including loading Meta AI's pretrained weights.
**Technology:** Core modern LLM components including **Rotary Position Embeddings (RoPE)**, **SwiGLU** activation functions, and **RMSNorm**.

## 08. Memory Efficient Weight Loading (`08_memory_efficient_weight_loading`)
**What is it?** A notebook demonstrating how to load massive model weights into RAM without crashing your system due to Out-Of-Memory (OOM) errors.
**Technology:** Efficient PyTorch `load_state_dict` techniques.

## 09. Extending Tokenizers (`09_extending-tokenizers`)
**What is it?** A from-scratch implementation of the GPT-2 BPE tokenizer.
**Technology:** **Byte-Pair Encoding (BPE)** algorithm (a sub-word tokenization method used by almost all modern LLMs).

## 10. LLM Training Speed Tips (`10_llm-training-speed`)
**What is it?** Shows how to push PyTorch to its limits for faster training.
**Technology:** PyTorch profiling, `torch.compile` for graph compilation, mixed-precision training (FP16/BF16), and hardware optimization.

## 11. Qwen3 Implementation (`11_qwen3`)
**What is it?** A complete from-scratch implementation of Qwen3 0.6B and the larger 30B-A3B.
**Technology:** **Mixture-of-Experts (MoE)** architecture, and handling multiple model variants (base, reasoning, coding).

## 12. Gemma 3 Implementation (`12_gemma3`)
**What is it?** A from-scratch implementation of Google's Gemma 3 270M model.
**Technology:** Advanced transformer architectures with KV caching support and Sliding Window Attention.

## 13. Olmo 3 Implementation (`13_olmo3`)
**What is it?** A from-scratch implementation of Ai2's Olmo 3 7B and 32B models.
**Technology:** Includes Base, Instruct, and "Think" (reasoning) variants, alongside efficient KV cache handling.

## 14. Chapter 5 With Other LLMs (`14_ch05_with_other_llms`)
**What is it?** Notebooks that swap out the standard GPT-2 model used in Chapter 5 with other models like Qwen3 and Llama 3.
**Technology:** Model interoperability and architecture swapping.

## 15. Tiny Aya 3.35B (`15_tiny-aya`)
**What is it?** A from-scratch implementation of Cohere's highly capable multi-lingual Tiny Aya model.
**Technology:** Features **Parallel transformer blocks** (computes attention and MLP from the same normalized input in one step to improve throughput), **Sliding Window Attention** (3:1 local to global ratio), and a modified **LayerNorm** without bias.
**References:**
- [Cohere Tiny Aya Announcement](https://cohere.com/blog/cohere-labs-tiny-aya)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

## 16. Qwen3.5 Implementation (`16_qwen3.5`)
**What is it?** A from-scratch style implementation of the Qwen3.5-0.8B model.
**Technology:** Based on the **Qwen3-Next architecture** which alternates between highly efficient **linear attention** layers and full attention layers.
**References:**
- [Beyond Standard LLMs (Linear Attention Hybrids)](https://magazine.sebastianraschka.com/p/beyond-standard-llms)

## 17. Gemma 4 Implementation (`17_gemma4`)
**What is it?** A from-scratch implementation of the newer Gemma 4 models (specifically the E2B and E4B dense variants).
**Technology:** Advanced dense LLM architecture.

## 18. Muon Optimizer (`18_muon`)
**What is it?** A guide on swapping standard optimizers (like AdamW) with the Muon optimizer during GPT training.
**Technology:** **Muon Optimizer** (a momentum-based orthogonalized optimizer designed to accelerate training and improve neural network generalization).
