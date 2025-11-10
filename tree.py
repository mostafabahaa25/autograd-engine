import os

def pretty_tree(dir_path, prefix='', ignore=('__pycache__', '.git', 'venv')):
    entries = [e for e in sorted(os.listdir(dir_path)) if e not in ignore]
    for index, name in enumerate(entries):
        path = os.path.join(dir_path, name)
        connector = '└── ' if index == len(entries) - 1 else '├── '
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = '    ' if index == len(entries) - 1 else '│   '
            pretty_tree(path, prefix + extension, ignore)

# Example usage
pretty_tree(r"E:\dataset\pro\autograd")
