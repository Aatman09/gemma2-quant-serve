In qunatiztion the weights are loaded in the memory in their quantized form and just when there is going to be a matrix multiplication , the weights are muplited with their scaling factor and converted to bfloat16 or bfloat32 and then the matrix multplcation happens and then the output is generated

float(x) can only covert one value to float 32 while x.float() coverts all elements of the tensor to float it is especially used in pytorch 


we cannot qunatize the weigths by just casting them to integer value this is beacause in pytorch when we do torch.tensor(x , dtype = torch.int8) it does integer overflow not integer clipping . so instead of rounding the numbers like 300 to 127 (which) is the limit of int8 what pytorch does it that it take a modulo with the total numbers i.e the output of 300 will be 300 % 255  which is 44 . 

Due to this the meaning and the value of the weights are lost 



That's per-tensor quantization. Per-channel is the whole reason int8 quantization stays accurate on LLM weights — different rows have wildly different magnitudes, and one global scale wastes range on every row. This is the core math of the project; get it right.


max() returns the max + indices
amax() returns only the max


Scalling is depened on channel/feature/each row and not on the full vector so according should not do is find the abs max of the full vector and then apply the scalling rule to this , instead we need to find the scaling factor for each row and then divide this 

