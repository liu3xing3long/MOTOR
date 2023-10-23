import pickle
import json


def main():
    root = '../dataset/VQA_RAD/'
    data_sets = ['train', 'test']

    # with open(fr'{root}/cache/trainval_ans2label.pkl', 'rb') as fp:
    #     ans2label = pickle.load(fp)
    #
    # with open(fr'{root}/cache/trainval_label2ans.pkl', 'rb') as fp:
    #     label2ans = pickle.load(fp)

    close_label2ans = []
    open_label2ans = []
    for ds in data_sets:
        output_json = []

        targets_by_qid = {}

        with open(fr'{root}/cache/{ds}_target.pkl', 'rb') as fp:
            targets = pickle.load(fp)
            for tt in targets:
                targets_by_qid[tt['qid']] = tt

        with open(fr'{root}/cache/trainval_label2ans.pkl', 'rb') as fp:
            trainval_label2ans = pickle.load(fp)

        with open(fr'{root}/{ds}set.json', 'rb') as fp:
            json_meta = json.load(fp)

            d_open_cache = []
            d_close_cache = []
            for jm in json_meta:
                this_json = {}
                for k, v in jm.items():
                    this_json[k] = v

                # "answer": "No",
                # "answer_type": "CLOSED",
                at = this_json['answer_type']
                qid = this_json['qid']
                qidlabel = targets_by_qid[qid]['labels']
                if len(qidlabel) == 0:
                    print(this_json)
                    print(targets_by_qid[qid])
                    continue

                qanswer = trainval_label2ans[qidlabel[0]]

                if at == 'CLOSED':
                    if qanswer not in close_label2ans:
                        close_label2ans.append(qanswer)
                        idx_answer = len(close_label2ans) - 1
                    else:
                        idx_answer = close_label2ans.index(qanswer)
                else:
                    if qanswer not in open_label2ans:
                        open_label2ans.append(qanswer)
                        idx_answer = len(open_label2ans) - 1
                    else:
                        idx_answer = open_label2ans.index(qanswer)

                this_json['answer'] = {
                    'answer': qanswer,
                    'labels': [idx_answer],
                    'scores': targets_by_qid[qid]['scores']}

                output_json.append(this_json)

            with open(fr'{root}/{ds}set_new.json', 'w') as fp:
                json.dump(output_json, fp, indent=4)

    with open(fr'{root}/cache/close_label2ans.pkl', 'wb') as fp2:
        pickle.dump(close_label2ans, fp2)
    with open(fr'{root}/cache/open_label2ans.pkl', 'wb') as fp2:
        pickle.dump(open_label2ans, fp2)

    print(open_label2ans)
    print(close_label2ans)
    print(len(open_label2ans))
    print(len(close_label2ans))


def check_pickle():
    datasets = ['open', 'close']

    for ds in datasets:
        dd_key_check = []
        d_out = []
        with open(fr'../dataset/VQA_RAD/cache/{ds}_label2ans_bkup.pkl', 'rb') as fp:
            d = pickle.load(fp)
            for dd in d:
                for k, v in dd.items():
                    if k in dd_key_check:
                        continue
                    else:
                        dd_key_check.append(k)
                        d_out.append({k: v})

        with open(fr'../dataset/VQA_RAD/cache/{ds}_label2ans.pkl', 'wb') as fp:
            print(d_out)
            print(fr'dataset {ds}, length {len(d_out)}')
            pickle.dump(d_out, fp)


if __name__ == '__main__':
    main()
    # check_pickle()
