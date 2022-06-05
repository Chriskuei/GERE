import json


dataset = [json.loads(line) for line in open('./data/fever_pipeline/dev.doc.json')]

save_file = open(f'data/dev.all.index.json', 'w')

for doc in dataset:
    evidences = doc['evidence']
    if doc['label'] == 'NOT ENOUGH INFO' and evidences != []:
        evidences = [evidences[0]]
    
    id_to_evidence = {}
    titles_set = set()
    titles_length = {}
    titles_index = {}
    for evidence in evidences:
        title, index, sentence, lines = evidence[0], evidence[1], evidence[2], evidence[-1]
        id_to_evidences = [line.split("\t")[:2] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
        titles_set.add(title)
        titles_length[title] = len(id_to_evidences)
    
    current_length = 0
    for title in list(titles_set):
        titles_index[title] = current_length
        current_length = current_length + titles_length[title]

    evidence_dict = {}
    all_id_to_evidence = {}
    valid_evidence_id = []
    for evidence in evidences:
        title, index, sentence, lines = evidence[0], evidence[1], evidence[2], evidence[-1]
        id_to_evidences = [line.split("\t")[:2] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
        map_num = {int(item[0]): idx+titles_index[title] for idx, item in enumerate(id_to_evidences) if item[0] != ''}

        id_to_evidence = {idx+titles_index[title]: item[1] for idx, item in enumerate(id_to_evidences) if item[0] != ''}
        all_id_to_evidence = {**all_id_to_evidence, **id_to_evidence}
        valid_evidence_id.append(map_num[index])
    doc['target_evidence'] = valid_evidence_id
    # evidence_dict['target_evidence'] = ' '.join(valid_evidence_id)
    # evidence_dict['target_evidence_bpe'] = self.encode(evidence_dict['target_evidence'])
    doc['evidence'] = all_id_to_evidence
    save_file.write(json.dumps(doc) + '\n')
