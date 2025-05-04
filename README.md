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
      - Modify /home/zhenlan/anaconda3/lib/python3.12/site-packages/unsloth_zoo/temporary_patches.py
         ```python        
            # if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
            # needed to ensure that the attention mask is the same size as the key and value states
            if attention_mask is not None:
                  seq_len = attention_mask.shape[-1]
                  key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]
         ```