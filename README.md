# Transformer & Universal Transformer

Transformer ('Attention is all you need' A. Vaswani et el.) &
Universal Transformer ('Universal Transformer' M. Dehghani et. el)

## Prerequisites

- python 3.6
- torch 1.2
- tensorboard 1.14+
- CUDA 9

## Data

- Source / target files: one sentence per line
- Source / target vocab files: one vocab per line, the top 5 fixed to be `<pad> <unk> <s> </s> <spc>` as defined in `utils/config.py`

## Train

To train the model - check `Examples/train.sh`

- `train_path_src` - path to source file for training
- `train_path_tgt` - path to target file for training
- `dev_path_src` - path to source file for validation (default set to `None`)
- `dev_path_tgt` - path to target file for validation (default set to `None`)
- `path_vocab_src` - path to source vocab list
- `path_vocab_tgt` - path to target vocab list
- `use_type` - `word` or tokenise into `char`
- `save` - dir to save the trained model
- `random_seed` - set random seed
- `share_embedder` - share embedding matrix across source and target
- `embedding_size_enc` - source embedding size
- `embedding_size_dec` - target embedding size
- `load_embedding_src` - load pretrained src embedding if provided
- `load_embedding_tgt` - load pretrained target embedding if provided
- `num_heads` - number of self attention heads (base - `8`)
- `dim_model` - transformer hidden state size (base - `512`)
- `dim_feedforward` - feed forward dimension (base - `2048`)
- `enc_layers` - number of encoder layers
- `dec_layers` - number of decoder layers
- `transformer_type` - transformer type `universal | standard` (`universal` shares params across layers)
- `max_seq_len` - maximum sequence length, longer sentences filtered out in training
- `batch_size` - batch size
- `seqrev` - train seq2seq in reverse order
- `eval_with_mask` - compute loss on non `<pad>` tokens (default `True`)
- `dropout` - dropout rate
- `embedding_dropout` - embedding dropout rate
- `num_epochs` - number of epochs
- `use_gpu` - set to `True` if GPU device is available
- `learning_rate_init` initial learning rate (default `0.0001`)
- `learning_rate` peak learning rate (default `0.0001`)
- `lr_warmup_steps` learning rate warm-up steps, no warm-up when set to `0` (default `0`, set `learning_rate_init == learning_rate`)
- `max_grad_norm` - gradient clipping
- `checkpoint_every` - number of batches trained for 1 checkpoint saved (if `dev_path*` not given, save after every epoch)
- `print_every` - number of batches trained for train losses printed
- `max_count_no_improve` - used when `dev_path*` is given, number of batches  trained (with no improvement in accuracy on dev set) before roll back
- `max_count_num_rollback` - reduce learning rate if rolling back for multiple times
- `keep_num` - number of checkpoint kept in model dir (used if `dev_path*` is given)
- `normalise_loss` - normalise loss on per token basis
- `minibatch_split` - if OOM, split batch into minibatch (note gradient descent still is done per batch, not minibatch)

## Test

To test the model - check `Examples/translate.sh`

- `test_path_src` - path to source text
- `seqrev` - translate in reverse order or not
- `path_vocab_src` - be consistent with training
- `path_vocab_tgt` - be consistent with training
- `use_type` - be consistent with training
- `load` - path to model checkpoint
- `test_path_out` - path to save the translated text
- `max_seq_len` - maximum translation sequence length (set to be at least larger than the maximum source sentence length)
- `batch_size` - batch size in translation, restricted by memory
- `use_gpu` - set to `True` if GPU device is available
- `beam_width` - beam search decoding
- `eval_mode` - default `1` (other modes for debugging)
