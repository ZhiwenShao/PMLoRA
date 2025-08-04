# PMLoRA
PMLoRA: Progressive and Masked Low-Rank Adaptation for Facial Action Unit Detection

# Future-Temporal Cross Attention (FTCA) and Feature Evolution Matrix (FEM)

## FTCA
In the FTCA module, we split the predicted sequence features along the temporal axis:
```
    x_past = x[:, :, :half_t]   # F_near
    x_future = x[:, :, half_t:] # F_distant
```
    1. F_distant is used as the Query, and F_near is used as the Key and Value in a cross-attention mechanism.
    2. The attention output goes through a gated fusion step to merge near-future and distant-future temporal information.
    3. Finally, the processed F_distant and F_near features are concatenated to form the final output.


## FEM
The FEM module processes near-future frames as follows:
    1. Pass near-future frame features through a filtering module (class FilterBlockProj) .
    2. Use these filtered features to compute a similarity matrix (def FEM_MATRIX), which is then used to build the evolution matrix.
    3. This evolution matrix serves as context, and is used in a cross-attention operation (class FEM) with distant-future features.
    
This helps the model propagate structural information from near-future features to improve the consistency of distant-future predictions.

##
Both FTCA and FEM modules can be directly integrated into the 3D U-Net architecture of diffusion-based models, enhancing the temporal modeling and structural consistency of precipitation predictions.
