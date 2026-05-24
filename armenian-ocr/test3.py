import json


def sort_text_lines_by_coordinates(text_lines, y_tolerance=5):
    """
    Sorts a list of text lines (each with 'text', 'x', 'y' attributes)
    by their coordinates to achieve a top-to-bottom, left-to-right reading order.
    """
    if not text_lines:
        return []

    # Sort all lines by y-coordinate first to facilitate row grouping
    text_lines.sort(key=lambda line: line['y'])

    sorted_text = []
    current_row = []
    
    for line in text_lines:
        if not current_row:
            current_row.append(line)
        else:
            # Check if the current line belongs to the same row as the previous one
            if abs(line['y'] - current_row[-1]['y']) < y_tolerance:
                current_row.append(line)
            else:
                # New row, process the previous one
                current_row.sort(key=lambda l: l['x']) # Sort by x for left-to-right
                sorted_text.extend([l['text'] for l in current_row])
                current_row = [line] # Start a new row

    # Process the last row
    if current_row:
        current_row.sort(key=lambda l: l['x'])
        sorted_text.extend([l['text'] for l in current_row])

    return sorted_text

with open('test1', 'r') as file:
    for ln, line in enumerate(file, 1):
        boxes = json.loads(line)
ll=[]
for box  in boxes[0]:
   dboxes={}
   dboxes['text']=box['text']
   dboxes['x']=box['box'][2]
   dboxes['y']=box['box'][3]
   ll.append(dboxes)

   
   #print(calc(box['box'][1],box['box'][0]))

# Example usage (assuming 'text_lines' is a list of dictionaries like {'text': '...', 'x': ..., 'y': ...})
sorted_content = sort_text_lines_by_coordinates(ll)
print(" ".join(sorted_content).replace("- ",""))