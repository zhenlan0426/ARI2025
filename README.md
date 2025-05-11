#### ARI2025 - Lessons
1. Based on Fine_Tune copy 2, 3, 5. mask does not seem to matter. branch off from functions_old
2. BOS_Y -> row -> col does not work well as model predict row and then proceed to predict cell 1 as shown in
   Fine_Tune copy 4. row_special_token -> row and col_special_token -> col
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
4. I am trying to solve ARC AGI problem using VLM. each task is represented as (input1, output1),(input2, output2),... (inputN, outputN), where input and output are 2d grid of integer between 0 and 9. Our goal is to given all history up till inputN, and predict outputN. 

5. StaticCache Modifications
   - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/transformers/cache_utils.py (line 1276). Together with cache_position will ensure
      DFS properly backtrack since Cache is not a copy.
      ```python        
         return k_out[:, :, :cache_position[-1].item()+1], v_out[:, :, :cache_position[-1].item()+1]
         ```
6. 4d position based attention mask. _update_causal_mask will ingore 4d attention mask passed in.
   - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py (line 546)
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