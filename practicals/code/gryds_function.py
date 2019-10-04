def gryds_function(images):
    import numpy as np
    import gryds
    new_image=[]
    
    for i in range(len(images)):


        affine = gryds.AffineTransformation(
                     ndim=2,
                     angles=[np.pi/4.], 
                     center=[0.5, 0.5]  
                     )
        
        random_grid = np.random.rand(2, 3, 3)
        random_grid -= 0.5
        random_grid /=10
        
        bspline = gryds.BSplineTransformation(random_grid) 

        interpolator = gryds.MultiChannelInterpolator(images[i], order=0, cval=[.1, .2, .3],mode='nearest')
        transformed_image = interpolator.transform(bspline,affine)
        new_image.append(transformed_image)
        
    return np.array(new_image)