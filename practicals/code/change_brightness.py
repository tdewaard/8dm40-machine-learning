def change_brightness(images):
    
    #input; (images): The input is a array of images 
    #output: (alterdImages): The output is an array of images from which the brightness has been randomly scaled with a value between 0.1 and     2.0

    import random 
    import numpy as np
    alterdImages=[]
    
    for i in range(len(images)):
        #generate a random value between 0.1 and 2.0 
        random_value=random.randrange(10,200)/100
        
        #multiply the entire image with the randomly generated value in order to scale the intensity
        changed_brightness=images[i]*random_value
        alterdImages.append(np.clip(changed_brightness,0,1))
    
    return np.array(alterdImages)