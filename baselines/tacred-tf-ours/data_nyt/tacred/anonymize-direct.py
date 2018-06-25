from stanza.text.dataset import Dataset

for fin_name in ['train.conll', 'test.conll']:
    fout_name = fin_name.replace('.conll', '.anon-direct.conll')
    print('loading {}'.format(fin_name))
    d = Dataset.load_conll(fin_name)
    print(d)
    for i, row in enumerate(d):
        for j in range(len(row['word'])):
            if row['subj'][j] == 'SUBJECT':
                d.fields['word'][i][j] = row['subj_ner'][j] + '-SUBJ'
            if row['obj'][j] == 'OBJECT':
                d.fields['word'][i][j] = row['obj_ner'][j] + '-OBJ'
    d.write_conll(fout_name)
