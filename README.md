### ARI2025 - ARC
Problem: To solve ARC AGI problem using LLM. each task is represented as (input1, output1),(input2, output2),... (inputN, outputN), where input and output are 2d grid of integer between 0 and 9. Our goal is to given all history up till inputN, and predict outputN.
#### Custom attention mask
1. Based on Fine_Tune copy 2, 3, 5. mask does not seem to matter. branch off from functions_old
2. BOS_Y -> row -> col does not work well as model predict row and then proceed to predict cell 1 as shown in
   Fine_Tune copy 4. row_special_token -> row and col_special_token -> col
#### VLM
3. VLM modifications:
      - position embedding
      - processing
         - image placeholder numbers
         - image size (processor.image_processor.preprocess(input_image,size=(64,64)))
         - image normalization
      - text embedding re-map
      - Modify modeling_siglip.py for position_ids to work with custom image size
         ```python
         - embeddings = embeddings + self.position_embedding(get_position_ids(height, width))
         ```

      - Modify modeling_gemma3.py
         - Gemma3MultiModalProjector (removed avg pooling)
         - self.get_image_features to work with list of images
         ```python        
               image_features = [self.get_image_features(pixel_value) for pixel_value in pixel_values] # 1, l_i, h
               image_features = torch.cat(image_features, dim=1) # 1, l, h
         ```

         - disable sliding window attention (comment out line 389 code block)
         ```python
            if self.is_sliding and attention_mask is not None:...
         ```

         - pass in 4d causal attention mask to disable cross attention among images
      - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/unsloth_zoo/temporary_patches.py for cache position to work.
         ```python        
            # if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
            # needed to ensure that the attention mask is the same size as the key and value states
            if attention_mask is not None:
                  seq_len = attention_mask.shape[-1]
                  key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]
         ```
      - Gemma3 images will attend to all other images even when injected in between "text". This will break causality for ARC problem.
      - More importantly, the vision tower is nothing more than a bi-directional transformer, similar to the LLM. and the input to the vision tower "color" contains the same information as the "text" with superficial format (image is arbitrary RGB and text is 0 ~ 9 token). Essentially, processing the same information twice. CNN based U-net might be a better choice to capture the spatial information.
5. StaticCache Modifications
   - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/transformers/cache_utils.py (line 1276). Together with cache_position will ensure
      DFS properly backtrack since Cache is not a copy.
      ```python        
         return k_out[:, :, :cache_position[-1].item()+1], v_out[:, :, :cache_position[-1].item()+1]
         ```
#### Relative Position Bias
6. 4d position based additive attention score. _update_causal_mask will ingore 4d attention mask passed in.
   - disable _update_causal_mask in /home/zhenlan/anaconda3/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py (line 546)
      ```python
         # causal_mask = self._update_causal_mask(
         #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
         # )
         causal_mask = attention_mask
      ```
   - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py (line 244)
      ```python
         # attention_mask[self.layer_idx:self.layer_idx+1],
         attn_output, attn_weights = attention_interface(
                                                         self,
                                                         query_states,
                                                         key_states,
                                                         value_states,
                                                         self.multi_grid_attention(self.layer_idx), # 4d attention mask
                                                         dropout=0.0 if not self.training else self.attention_dropout,
                                                         scaling=self.scaling,
                                                         sliding_window=self.sliding_window,
                                                         **kwargs,
                                                   )
      ```
      attention_mask is of shape (layers, heads, seq_len, seq_len) from MultiGridAttention. only works when batch size is 1
- Tried seperate parameter for different layers or same parameter for all layers. No difference in performance. **grad is really small (1e-8)**. model performance is worse when RoPE is disabled (due to difference from pretrain?) - **Fine_Tune copy 12**
- Implement custom backward pass for relative position bias (MultiGridAttention2) for better VRAM management.
- Future work: 
   - test without RoPE and disable input only transformation to ease learning.
   - parametrize attention score with MLP(relative row and col) -> bias
#### 2d RoPE
- introduce Qwen3RotaryEmbedding2d and rows, cols position embedding (Fine_Tune copy 14)
- Qwen3RotaryEmbedding2d features different frequencies for different heads. need to have cos.repeat_interleave due to 32 heads for query but only 8 for key,value.
- apply_rotary_pos_emb needs to be modified to use cos_expanded and sin_expanded for query, cos and sin for key.
- tokenize_causal introduces offset1 and offset2 for model to tell the difference between input and output of same example and between different examples.
- take longer to train with worse performance. Most likely due to different rotation from pretrain domain.
- trainable head_freq is not working as pytorch complains about backward pass for a second time. Most likely due to cos and sin are shared across layers.
```python
class Qwen3RotaryEmbedding2d(nn.Module):
    def __init__(self, config: Qwen3Config, max_head_freq=100, device=None):
        super().__init__()
        ...
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        mask = torch.arange(len(inv_freq)) % 2 == 0
        freq_x = inv_freq.clone()
        freq_y = inv_freq.clone()
        freq_x[~mask] = 0.0
        freq_y[mask] = 0.0
        # different freq for different heads, fast forward position as theta = 1000000, much bigger than seq_len
        self.head_freq = torch.linspace(1, max_head_freq, 8) # 8 groups for key,value

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        rows, cols = position_ids # (seq_len)
        rows = rows[None, :, None].float() # (1, seq_len, 1)
        cols = cols[None, :, None].float() # (1, seq_len, 1)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freq_x = self.freq_x.float()[None, None, :].to(x.device) # (1, 1, half_dim)
            freq_y = self.freq_y.float()[None, None, :].to(x.device) # (1, 1, half_dim)
            angles = rows * freq_x + cols * freq_y # (1, seq_len, half_dim)
            emb = torch.cat((angles, angles), dim=-1)[:, None] # (1, 1, seq_len, hidden_dim)
            emb = self.head_freq[None,:, None, None].to(x.device) * emb # (1, heads, seq_len, hidden_dim)
            cos = emb.cos() * self.attention_scaling # (1, 8, seq_len, hidden_dim) for key, value
            sin = emb.sin() * self.attention_scaling
            cos_expanded = cos.repeat_interleave(4, dim=1)  # → (1,32,seq_len,dim) for query
            sin_expanded = sin.repeat_interleave(4, dim=1)  # → (1,32,seq_len,dim)
        return cos_expanded.to(dtype=x.dtype), sin_expanded.to(dtype=x.dtype), cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```
#### enriched feature
- unlike token in NLP, cell value of 0 ~ 9 is not semantically meaningful by itself, its value is entirely dependent on the context.
- hand crafted features are described in features.md and implemented in features_optimized.py. There are 2 versions of features, causal and non-causal, i.e., features dependent on
current or prior cells in flattened grid or any cell in the grid. Only causal features can be used for autoregressive training, input features are non-causal.
- extract_features takes in grid and background color (0) and returns a flattened feature vector of shape (n_pixels, n_features).
- tokenize_features takes in a task (list of input and output grid) and returns a input_tokens (with placeholder for features injections), and features (total flattened sequence length, n_features). 
input_tokens: BOS_X, placeholder_token * flattened sequence length, EOS_X, BOS_Y, placeholder_token * flattened sequence length, EOS_Y...
- FeatureEmbedding is used to map features to embedding space and then injected into the placeholder_token in input_tokens for LLM to consume.

#### Model Target
- autoregressive on all (all causal) -> autoregressive on output (input non-causal, output causal) -> autoregressive on outputN / one-shot on outputN
- less training signal as we autoregressively train on less target but with richer features. **one-shot on outputN does not work well and size prediction does not work well as shown in Fine_Tune copy 19**

#### Input representation
- Conceptually, the problem is of dimension (example_count, input / output, height, width, color_channel) and we flatten the first 4 dimensions for llm to process. The flattened input loses the original
structure (line break, EOS_X, EOS_Y, BOS_X, BOS_Y are the only ways in theory to recover the original structure for the llm).
- might be easier to explicitly represent the input as sum of example_id, input / output, position_id (row, col) and color_channel.
- Implemented in CombinedEmbedding
- Randomly permute example_id, to ensure all ids are trained and what exact example_id is not important but different ids are to be treated as different examples.
   ```python
   example_permutation = np.random.permutation(30)
   example_ids.extend([example_permutation[i]] * (input_end_len - input_start_len))    
   ```
