import argparse
import os
import tkinter as tk
from models.app import Application 





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_path','-p',type=str,help='Path to the dataset')
    argparser.add_argument('--device','-d',type=str,default='cuda:0',help='Device to run the model on, gpu will accelerate the rendering process')
    args = argparser.parse_args()
    root = tk.Tk()
    app = Application(root, args.dataset_path, args.device)
    root.mainloop()
