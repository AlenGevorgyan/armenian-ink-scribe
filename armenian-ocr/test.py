import json
ls={}
with open('output', 'r') as file:
    for ln, line in enumerate(file, 1):
        result = json.loads(line)
        #i=0
        for k in result[0]:
            #print(k['box'][0])
            #Arr=k['box'].split(",")
            ls[k['box'][0]]=k['text']
            #print(result[0][k]['text'])
            #print(k['text'])
            #i+=1

#ll=sorted(ls)
ll = dict(sorted(ls.items()))
for key in ll:
    value = ll[key]
    print(f"{key}: {value}")

    #print(l[]