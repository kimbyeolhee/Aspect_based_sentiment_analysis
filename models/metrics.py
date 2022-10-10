from sklearn.metrics import f1_score


def evaluate(y_true, y_pred, label_len):
    count_list = [0] * label_len
    hit_list = [0] * label_len

    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1

    for i in range(len(count_list)):
        if count_list[i] == 0:
            count_list[i] += 1

    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i] / count_list[i])

    accuracy = sum(hit_list) / sum(count_list)
    macro_accuracy = sum(acc_list) / 3

    print("accuracy: ", accuracy)
    print("macro_accuracy: ", macro_accuracy)

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    eval_f1_score = f1_score(y_true, y_pred, average=None)
    f1_score_micro = f1_score(y_true, y_pred, average="micro")
    f1_score_macro = f1_score(y_true, y_pred, average="macro")

    print("f1_score: ", eval_f1_score)
    print("f1_score_micro: ", f1_score_micro)
    print("f1_score_macro: ", f1_score_macro)

    return accuracy
