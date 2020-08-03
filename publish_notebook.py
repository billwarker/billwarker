# get notebook from command line argument
# generate title, front matter, tags
# run nb convert
# if output is a file move it to posts
# if output is a directory rename the md file to index.md
# and move the dir to posts
# commit to github

# this file has the alias pnb in the zshrc file

import os
import shutil
import re
import fileinput
import sys

def parse_notebooks_from_cmd_args(args):
    args = args[1:]
    if '.' in args:
        # publish every notebook in the dir
        print('YEET')
    else:
        notebook_files = []
        for arg in args:
            if arg.endswith('.ipynb'):
                notebook_files.append(arg)
        
        return notebook_files

def fix_image_links(file="index.md", images_dir="images"):
    
    def replaceAll(file, searchExp, replaceExp):
        for line in fileinput.FileInput(file, inplace=1):
            if searchExp in line:
                line = line.replace(searchExp,replaceExp)
            sys.stdout.write(line)
            
    with open(file, 'r') as f:
        filedata = f.read()
        # Find all markdown link syntaxes
        md_links = re.findall('!\\[[^\\]]+\\]\\([^)]+\\)', filedata)
        for link in md_links:
            new_link = re.sub(r'\(.*\/', f'({images_dir}/', link)
            replaceAll(file, link, new_link)
            
def convert_notebook_to_md(notebook):

    # make new directory
    root = notebook.replace('.ipynb', '')
    if root not in os.listdir():
        os.mkdir(root)
    shutil.copy(notebook, os.path.join(root, notebook))
    
    # delete old directory, convert to md inside new directory
    os.chdir(root)
    os.system(f"jupyter nbconvert --to markdown '{notebook}'")
    
    # rename output to index.md
    os.rename(root + ".md", "index.md")
    
    # rename images folder if it exists
    images_dir = root + "_files"
    if images_dir in os.listdir():
        os.rename(images_dir, "images")
        fix_image_links(file="index.md", images_dir="images")

    # delete original notebook
    os.remove(notebook)

    return root

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


if __name__ == "__main__":
    note_path = os.getcwd()
    posts_path = os.path.join(os.path.dirname(__file__),
                              os.path.join("content", "posts"))
                        
    notebooks = parse_notebooks_from_cmd_args(sys.argv)
    for notebook in notebooks:
        note_dir = convert_notebook_to_md(notebook)
        dest = os.path.join(posts_path, note_dir)
        os.mkdir(dest)
        copytree(os.getcwd(), dest)
        shutil.rmtree(os.path.join(note_path, note_dir))