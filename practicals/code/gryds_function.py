def gryds_function(images):
    # input: (images): The input is an array of images  
    # output: (new_image): The output is a  geometricly augmented array of images with a randomly defined deformation/ rotation.
    
    import numpy as np
    import gryds
    
    new_image=[]
    
    # A unique geomatrical augmentation is generated and applied to each image from the input
    for i in range(len(images)):
        
        # Defining the rotation operator
        affine = gryds.AffineTransformation(
                     ndim=2,
                     angles=[np.pi/4.], 
                     center=[0.5, 0.5]  
                     )
        
        # Defining the different deformation operator
        random_grid = np.random.rand(2, 3, 3)
        random_grid -= 0.5
        random_grid /=10
       
        bspline = gryds.BSplineTransformation(random_grid) 
        interpolator = gryds.MultiChannelInterpolator(images[i], order=0, cval=[.1, .2, .3],mode='nearest')
        
        # Apply the deformation matrices to the input images
        transformed_image = interpolator.transform(bspline,affine)
        new_image.append(transformed_image)
        
    return np.array(new_image)