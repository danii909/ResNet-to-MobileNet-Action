def extract(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    # Find first ''' and last '''
    start = text.find("'''") + 3
    end = text.rfind("'''")
    return text[start:end]

c1 = extract('scratch/write_v7_base.py')
c2 = extract('scratch/append_slides1.py')
c3 = extract('scratch/append_slides2.py')
c4 = extract('scratch/append_slides3.py')

with open('scratch/make_presentation_v7.py', 'w', encoding='utf-8') as f:
    f.write(c1 + c2 + c3 + c4)
