

def rescaling(scaling_factor, rescale, Y):
    if rescale == 1:
        if scaling_factor == None:
            raise NotImplementedError("scaling factor must be specified")
        return (Y // scaling_factor)
    elif rescale == 0:
        return Y
    else:
        raise NotImplementedError("Rescale must be 0 or 1")
