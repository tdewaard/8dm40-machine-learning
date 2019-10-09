def gryds_function(images,segmentations):
    # input: (images): The input is an array of images  
    # input: (segmentations): The segmentations of the corresponding input images
    
    # output: (new_image): The output is a  geometricly augmented array of images with a randomly defined deformation/ rotation.
    # output:(new_segmentation): The output the augmented version of the segmentation 
    
    import numpy as np
    import gryds
    new_image=[]
    new_segmentation=[]
    for i in range(len(images)):


        affine = gryds.AffineTransformation(
                     ndim=2,
                     angles=[np.pi/4.], # List of angles (for 3D transformations you need a list of 3 angles).
                     center=[0.5, 0.5]  # Center of rotation.
                     )
        
        random_grid = np.random.rand(2, 3, 3)
        random_grid -= 0.5
        random_grid /=20
        
        bspline = gryds.BSplineTransformation(random_grid) 
        # define interpolator of the input_image
        interpolator = gryds.MultiChannelInterpolator(images[i], order=0, cval=[.1, .2, .3],mode='nearest')
        
        #define interpolator of the segmentation image
        interpolator_segentation = gryds.Interpolator(segmentations[i][:, :, 0],mode='constant')
        
        #transform the input image
        transformed_image = interpolator.transform(bspline,affine)
        
        #transform the segmentation image
        transformed_segmentation=interpolator_segentation.transform(bspline,affine)
        
        #add results into lists
        new_segmentation.append(np.clip(transformed_segmentation,0,1))
        new_image.append(np.clip(transformed_image,0,1))
        
    return np.array(new_image),np.array(new_segmentation)

