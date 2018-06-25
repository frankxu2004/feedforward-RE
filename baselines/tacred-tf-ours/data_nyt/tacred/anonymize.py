from stanza.text.dataset import Dataset

# for fin_name in ['train.conll', 'dev.conll', 'test.conll']:
#     fout_name = fin_name.replace('.conll', '.anon.conll')
#     print('loading {}'.format(fin_name))
#     d = Dataset.load_conll(fin_name)
#     print(d)
#     for i, row in enumerate(d):
#         if row['subj'] == 'SUBJECT':
#             d.fields['word'][i] = row['subj_ner']
#         if row['obj'] == 'OBJECT':
#             d.fields['word'][i] = row['obj_ner']
#     d.write_conll(fout_name)


for fin_name in ['train.conll', 'dev.conll', 'test.conll']:
    fout_name = fin_name.replace('.conll', '.anon.conll')
    print('loading {}'.format(fin_name))
    d = Dataset.load_conll(fin_name)
    print(d)
    for i, row in enumerate(d):
        for j in range(len(row['word'])):
            if row['subj'][j] == 'SUBJECT':
                d.fields['word'][i][j] = 'NER-' + row['subj_ner'][j]
            if row['obj'][j] == 'OBJECT':
                d.fields['word'][i][j] = 'NER-' + row['obj_ner'][j]
    d.write_conll(fout_name)
