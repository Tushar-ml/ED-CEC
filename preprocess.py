# from modules.embeddings import wordpiece_tokenizer, align_lists, assign_labels, combine_labels, tokenizer
# from modules.error_detection import get_bert_embeddings, token_positional_embeddings
# import torch.nn as nn
# import torch
# import torch.nn.functional as F

# prediction = "director did not ask for a catechism only."
# actual = "the rector did not ask for a catechism only."

# prediction_tokens = wordpiece_tokenizer(prediction)
# actual_tokens = wordpiece_tokenizer(actual)

# actual_list, prediction_list = align_lists(prediction_tokens, actual_tokens)
# labels = assign_labels(prediction_list, actual_list)
# pred, labels = combine_labels(prediction_list, labels)
# pred_embeddings = get_bert_embeddings(pred)

# # Error Detection Module
# index_of_change = [idx for idx, label in enumerate(labels) if label == "C"]

# # Context Aware Correction Module
# e_k = (pred_embeddings[index_of_change[0]])

# decode_inputs = 'the'

# token, positional = token_positional_embeddings(decode_inputs, return_embed = False)
# combine_tokens = token + positional
# combine_tokens = combine_tokens.squeeze(0)


# result = torch.concatenate([combine_tokens, e_k]).t()
# decode_embed = token_positional_embeddings(decode_inputs, True)

# fc_layer = nn.Linear(result.shape[1],decode_embed.shape[1])

# e_z = fc_layer(result)
# e_z = e_z.t().unsqueeze(0)

# print(e_z.shape)
# # Transformer Decoder
# decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=12, batch_first=True)
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

# decoder_layer_output = transformer_decoder(
#     tgt=e_z,
#     memory=pred_embeddings 
# )

# generation_output = F.softmax(decoder_layer_output, dim=-1).squeeze(0)


# # Rare Word list
# word_list = ["battleax", "rector"]
# word_list_embeds = []

# for word in word_list:
#     # tokens = tokenizer.tokenize(word)
#     embeds = get_bert_embeddings(word).mean(dim=1)
#     print("word embed shape: ", embeds.shape)
#     word_list_embeds.append(embeds)

# word_list_stack = torch.vstack(word_list_embeds)
# M_o = nn.Parameter(torch.randn(1, 768))

# mo_mc = torch.concatenate([M_o, word_list_stack]) 
# print(generation_output.shape,mo_mc.shape)

# scores_t = torch.matmul(generation_output, mo_mc.t())

# gate_t = scores_t[:, 0]
# m = torch.argmax(scores_t)

# def find_mth_index(m: torch.Tensor, scores_t: torch.Tensor) -> int:
#     m = m.item()
#     dim_ = scores_t.shape[0]
#     return m//dim_

# print(len(word_list_embeds), m, word_list_stack.shape)
# mth_index = find_mth_index(m, scores_t)

# e_m_c = word_list_embeds[max(0,mth_index-1)]

# o_con = F.softmax(torch.matmul(generation_output, e_m_c.t())/2)
# o_con = torch.matmul(o_con, e_m_c)
# print(e_m_c.shape, generation_output.shape, o_con)
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()

        # Define fully connected layers
        self.fc_gate = nn.Linear(input_size, 1)

    def forward(self, gen_probs, con_probs):
        # Apply sigmoid activation to the gate
        gate = torch.randn(54, 1)

        # Expand dimensions of gate to match the dimensions of gen_probs
        gate = gate.unsqueeze(-1)

        # Weighted sum along the third dimension
        weighted_sum = gate * gen_probs + (1 - gate) * con_probs

        return weighted_sum

# Example usage
input_size = 768  # Adjust based on the actual size of your input
model = CustomModel(input_size)

# Example generation probabilities and contextual probabilities (replace with actual values)
gen_probs = torch.randn(54, 6, 30522)
con_probs = torch.randn(54, 6, 30522)

# Forward pass
output = model(gen_probs, con_probs)

print("Weighted Sum Output Shape:", output.shape)
