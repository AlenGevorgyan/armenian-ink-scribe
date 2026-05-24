import json
def calc(x, y):
    # Example: Calculate the sum of squares
    return x**2 + y**2

with open('output', 'r') as file:
    for ln, line in enumerate(file, 1):
        boxes = json.loads(line)

dboxes={}
for box  in boxes[0]:
   k=calc(box['box'][1],box['box'][0])
   dboxes[k]=box['text']
   #print(calc(box['box'][1],box['box'][0]))
dd = dict(sorted(dboxes.items()))

for key, value in dd.items() :
    print (key, value)


#dd=sorted(dboxes)
#print(dd)
exit(0)
#dboxes=[]
#for box in boxes[0]:
#    dboxes.append(box['box'])
#    i+=1
dboxes=boxes[0]
print(dboxes)
vb = sorted(dboxes, key=lambda vbox: (vbox['box'][1]))
#print(vb)
#exit(0)

for box  in vb:
    #hb = sorted(box, key=lambda hbox: (hbox['box'][]))
    print(box['box'][0],box['text'])
    #print(hb[1], end=" ")
exit(0)

sb = sorted(dboxes, key=lambda box: (box['box'][1], box['box'][2]))
print(sb)
#exit(0)
#sorted_boxes_tb_lr = sorted(dboxes, key=lambda box: (box[0]['box'][1], box[0]['box'][0]))
#print(sorted_boxes_tb_lr)
#exit(0)
#sorted_boxes_tb_lr = sorted(boxes['box'], key=lambda box: (box[0]))
#print(f"Sorted top-to-bottom, then left-to-right: {sorted_boxes_tb_lr['text']}")

for f in sb:
    print(f['text'], end=" ")

#sorted_boxes_by_xmin = sorted(boxes, key=lambda box: box[0])
#print(f"Sorted by x_min: {sorted_boxes_by_xmin}")