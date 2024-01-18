from modules.embeddings import wordpiece_tokenizer, align_lists, assign_labels, combine_labels

prediction = "isite from chicago to denver"
actual = "what flights from chicago to denver"

prediction_tokens = wordpiece_tokenizer(prediction)
actual_tokens = wordpiece_tokenizer(actual)

actual_list, prediction_list = align_lists(prediction_tokens, actual_tokens)

labels = assign_labels(prediction_list, actual_list)

pred, labels = combine_labels(prediction_list, labels)

for p,l in zip(pred, labels):
    print(p,l)
# print(prediction_list, actual_list, labels)