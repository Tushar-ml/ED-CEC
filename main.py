
from modules.utils import align_lists, assign_labels


text = "director did not ask for a catechism"
truth = "the rector did not ask for a catechism"



list1 = tokenize(text)
list2 = tokenize(truth)


list2, list1 = align_lists(list1, list2)

labels = assign_labels(list1, list2)
print(list1, list2, labels)
# for idx in range(len(list1)):
#     print(f'{list1[idx]} {list2[idx]} => {labels[idx]}')