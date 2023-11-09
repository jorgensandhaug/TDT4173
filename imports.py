import nbformat

# Load the notebook
notebook_path = 'long_notebook_4_test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Extract all imports from the notebook's code cells
imports = set()
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        lines = cell['source'].split('\n')
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line)

# Combine all unique imports into a single string
import_block = '\n'.join(sorted(imports))

# Now you can manually add `import_block` to the top of your notebook
# Or you can use the nbformat library to insert a new cell at the top
new_cell = nbformat.v4.new_code_cell(source=import_block)
nb['cells'].insert(0, new_cell)

# Optionally, remove or comment out the original imports
# ...

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)