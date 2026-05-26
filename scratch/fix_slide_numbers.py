import re

with open('scratch/make_presentation_v7.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

slide_counter = 0
for i in range(len(lines)):
    line = lines[i]
    if line.startswith('# SLIDE '):
        slide_counter += 1
        lines[i] = re.sub(r'# SLIDE \d+', f'# SLIDE {slide_counter}', line)
    
    # Check for footer(sl, x) or footer(sl, x, y)
    if 'footer(sl,' in line:
        # Match footer(sl, num) or footer(sl, num, total)
        # We can just replace the first number with slide_counter
        lines[i] = re.sub(r'footer\(sl, \d+', f'footer(sl, {slide_counter}', line)
        if slide_counter == 23: # The last slide is 24, wait, total is 24. 
            pass # We will handle total slides globally

with open('scratch/make_presentation_v7.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Also fix the TOTAL_SLIDES
with open('scratch/make_presentation_v7.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'TOTAL_SLIDES = \d+', 'TOTAL_SLIDES = 24', text)
text = re.sub(r'footer\(sl, 24, \d+\)', 'footer(sl, 24, 24)', text)

with open('scratch/make_presentation_v7.py', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Total slides found: {slide_counter}")
