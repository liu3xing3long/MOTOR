import pickle
import json


def main():
    root = '../dataset/SLAKE/'
    data_sets = ['train', 'test']

    close_label2ans = []
    open_label2ans = []
    for ds in data_sets:
        output_json = []

        with open(fr'{root}/{ds}.json', 'rb') as fp:
            json_meta = json.load(fp)

            for jm in json_meta:
                this_json = {}
                for k, v in jm.items():
                    this_json[k] = v
                at = this_json['answer_type']
                qid = this_json['qid']
                qanswer = this_json['answer']
                qlang = this_json['q_lang']

                if qlang != 'en':
                    continue

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
                    'scores': [1.0]
                }

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
