import os

system_agnostic_path = lambda path: os.path.join("data", *path.encode("unicode_escape").decode().strip().split("\\"))

with open("data/training_fixed.txt") as f:
    for line in f:
        splitted = line.split("\t")
        if len(splitted) != 13:
            print(line)
        else:
            path_from = system_agnostic_path(splitted[9])
            path_to = system_agnostic_path(splitted[5])

            if os.path.exists(path_from):
                if os.path.exists(path_to):
                    print("ALREADY EXISTS", path_to)
                else:                       
                    dir_from, filen_from = os.path.split(path_from)
                    name_from, _ = os.path.splitext(filen_from)

                    dir_to, filen_to = os.path.split(path_to)
                    name_to, _ = os.path.splitext(filen_to)

                    os.makedirs(dir_to, exist_ok=True)

                    # move jpg
                    print("copy", path_from, path_to)
                    os.link(path_from, path_to)

                    # move json
                    print("copy", os.path.join(dir_from, name_from + ".json"), os.path.join(dir_to, name_to + ".json"))
                    os.link(os.path.join(dir_from, name_from + ".json"), os.path.join(dir_to, name_to + ".json"))

            else: 
                print("CANNOT MOVE", path_from)
            

# PROBLEMS:
                
# Not existing files in data/mandrac_imgs1
# they took 20 of 34 photos                
                
# both copied to the same location:
# DuplImagePair	61	0.05	Turn_0	None	Dataset\test\test_image2.JPG	1920 x 1080	JPG	280 KB	bistrina_imgs\record11_186_51.JPG	1920 x 1080	JPG	280 KB
# DuplImagePair	61	0.94	Turn_0	None	Dataset\test\test_image2.JPG	1920 x 1080	JPG	280 KB	bistrina_imgs\record11_180_63.JPG	1920 x 1080	JPG	318 KB
                
# DuplImagePair	61	0.94	Turn_0	None	Dataset\test\test_image38.JPG	1920 x 1080	JPG	318 KB	bistrina_imgs\record11_186_51.JPG	1920 x 1080	JPG	280 KB
# DuplImagePair	61	0.05	Turn_0	None	Dataset\test\test_image38.JPG	1920 x 1080	JPG	318 KB	bistrina_imgs\record11_180_63.JPG	1920 x 1080	JPG	318 KB
                

# if the 1st column is height?
                
