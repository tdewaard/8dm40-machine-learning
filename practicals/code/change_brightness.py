def change_brightness(images):
    import random 
    import numpy as np
    alterdImages=[]
    for i in range(len(images)):
        random_value=random.randrange(50,200)/100
        changed_brightness=images[i]*random_value
        alterdImages.append(changed_brightness)
    
    return np.array(alterdImages)