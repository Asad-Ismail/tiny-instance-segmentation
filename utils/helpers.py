




def fixed_points(x,y,mask):
    """Fixed Mask if centroid is not in the middle go to left to find one valid point"""
    h,w=mask.shape
    while x<w and mask[y][x]==0:
        x+=1
    return (y,x)

def get_quantized_center(x,y,mask,dst_size=64,p_sofar=None):
    h,w=mask.shape
    cy=y*(dst_size/h)
    cx=x*(dst_size/w) 
    # Quantized centers
    cyq=int(cy)
    cxq=int(cx)
    # Offsets of centers
    offy=cy-cyq
    offx=cx-cxq
    # Make sure two objects are not assigned same center
    if (cyq,cxq) in p_sofar:
        assert False
    p_sofar.add((cyq,cxq))
    return cyq,cxq,offy,offx,p_sofar

def find_center(mask):
    (ys,xs)=np.nonzero(mask)
    points=list(zip(xs,ys))
    h,w=mask.shape
    assert max(ys)<h and max(xs)<w
    #horizontal cucumber
    midx=None
    midy=None
    if abs(max(ys)-min(ys))<abs(max(xs)-min(xs)):
        xs=sorted(xs)
        midx=xs[len(xs)//2]
        yrel=[y for  x,y in points if x==midx]
        yrel=sorted(yrel)
        midy=yrel[len(yrel)//2]
    else:
        ys=sorted(ys)
        midy=ys[len(ys)//2]
        xrel=[x for  x,y in points if y==midy]
        xrel=sorted(xrel)
        midx=xrel[len(xrel)//2]
    return (midy,midx)    

def find_center_cv(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=mask.shape
    c = max(contours, key = cv2.contourArea)
    M = cv2.moments(c)
    #calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cY,cX)


