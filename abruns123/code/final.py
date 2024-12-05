"""
the final.py module includes importable functions
designed for different tasks relating to the final
"""
import os

def f_k(t):
    """
    f_k takes in a temperature value t and 
    returns that temperature in kelvin
    """
    return (t-32)*5/9+273.15

def parse(path):
    """
    parse takes in a file path as a string and 
    parses through the file. If it finds 
    Temperature: as the first word of a line,
    it will define the next string as the temperature.
    parse then returns this temperature.
    """
    lines=[]
    temp=0
    condition=False
    try:
        with open(path, "r",encoding='utf-8') as file:
            lines=[line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
    for line in lines:
        words=line.split()
        if words[0]=="Temperature:":
            temp=int(words[1])
            condition=True
    if condition is True:
        return temp

    return "No temperature in this file."

def md_list(path):
    md=[]
    files=os.listdir(path)
    for file in files:
        if file.endswith(".md"):
            md.append(file)
    folders=[item for item in files if os.path.isdir(os.path.join(path, item))]
    for folder in folders:
        new_path=path+"/"+folder
        new_md=md_list(new_path)
        for m in new_md:
            md.append(m)
    return md
    

print(md_list("/workspaces/CP1-24-final/CP1-24-final/abruns123"))