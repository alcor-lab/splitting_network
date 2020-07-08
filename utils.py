##check flag
import os 
import platform
def check_flag(args):

    if (args.dataPreprocessing):
        print("Dati già scalati a 256 e trasformati in png")
    else:
        print("Dataset Originale")
    if (args.baseline):
        if (args.mode_input == "rgb" or args.mode_input == "d"):
           print("Error: La modalità baseline è fatta solo per rgbd")
           os.sys.exit()
        else:
           print("Modalità baseline: prende in input Rgb e Depth e usa due reti BMVC concatenando le features")
    else:
        if(args.mode_input == "rgb"):
            print("Modalità Rgb: rete BMVC")
        if(args.mode_input == "d"):
            print("Modalità Depth: rete BMVC")
        if(args.mode_input == "rgbd"):       
            print("Modalità Rgbd: DepthMap")

def make_path(args):
    #check dataset / check machine  
    path_dataset_train = ""
    path_dataset_val = ""


    if (args.dataset == 'FPHA'):
        #### ARES ####
        if (platform.node() == "ares"): ##non ricordo su Aphrodite che path avevano...da aggiungere 
            if(args.dataPreprocessing):
                path_dataset_train = "/scratch/dataset/FPHA/train"
                path_dataset_val = "/scratch/dataset/FPHA/val"
            else:
                path_dataset_train = "/scratch/dataset/train"
                path_dataset_val = "/scratch/dataset/val"

        #### Cluster #####
        if (platform.node() == "frontend" or platform.node() == "node1" or platform.node() == "node2"): ##non ricordo su Aphrodite che path avevano...da aggiungere 
            if(args.dataPreprocessing):
                path_dataset_train = "/home/mirco/FPHA/train"
                path_dataset_val = "/home/mirco/FPHA/val"
            else:
                path_dataset_train = "/home/mirco/FPHA_Original/train"
                path_dataset_val = "/home/mirco/FPHA_Original/val"
	#### Add other Machine

        if (platform.node() == "--- "): ##non ricordo su Aphrodite che path avevano...da aggiungere 
            if(args.dataPreprocessing):
                path_dataset_train = "..."
                path_dataset_val = "..."
            else:
                path_dataset_train = "..."
                path_dataset_val = "..."



    if (args.dataset == 'GUN71'):   
        if (platform.node() == "ares"):
            if(args.dataPreprocessing):
                path_dataset_train = "..." #devi aggiungere il path del dataset trasformato 
                path_dataset_val = "..."
            else:
                path_dataset_train = "/scratch/dataset/GUN71/"
                path_dataset_val = "/scratch/dataset/GUN71/"
        if (platform.node() == "frontend" or platform.node() == "node1" or platform.node() == "node2"): ##non ricordo su Aphrodite che path avevano...da aggiungere  
            if(args.dataPreprocessing):
                path_dataset_train = "..." #devi aggiungere il path del dataset trasformato 
                path_dataset_val = "..."
            else:
                path_dataset_train = "/home/mirco/GUN71/"
                path_dataset_val = "/home/mirco/GUN71/"
        if (platform.node() == "--- "): ##non ricordo su Aphrodite che path avevano...da aggiungere 
            if(args.dataPreprocessing):
                path_dataset_train = "..."
                path_dataset_val = "..."
            else:
                path_dataset_train = "..."
                path_dataset_val = "..."
    
    
    if (args.dataset == 'SBU'):   
        if (platform.node() == "ares"):
            if(args.dataPreprocessing):
                path_dataset_train = "..." #devi aggiungere il path del dataset trasformato 
                path_dataset_val = "..."
            else:
                path_dataset_train = "..."
                path_dataset_val = "..."
        if (platform.node() == "frontend" or platform.node() == "node1" or platform.node() == "node2"): ##non ricordo su Aphrodite che path avevano...da aggiungere  
            if(args.dataPreprocessing):
                path_dataset_train = "..." #devi aggiungere il path del dataset trasformato 
                path_dataset_val = "..."
            else:
                path_dataset_train = "/home/paolo/datasets/SBU_Kinect_Interaction/exp" + str(args.split + 1) 
                path_dataset_val = "/home/paolo/datasets/SBU_Kinect_Interaction/exp" + str(args.split + 1)
        if (platform.node() == "--- "): ##non ricordo su Aphrodite che path avevano...da aggiungere 
            if(args.dataPreprocessing):
                path_dataset_train = "..."
                path_dataset_val = "..."
            else:
                path_dataset_train = "..."
                path_dataset_val = "..."
    
    
    
    
    ####Add other Dataset
    
    return path_dataset_train, path_dataset_val
