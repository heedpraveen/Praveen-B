data = [{'V': 'S001'}, {'V': 'S002'}, {'V': 'S009'}, {'VI': 'S001'}, {'VI': 'S005'}, {'VII': 'S005'}, {'VIII': 'S007'}]
uniq_list=[]
for i in data:
    if i not in uniq_list:
        uniq_list.append(i)
for i in uniq_list:
    print(set(i.values()))
