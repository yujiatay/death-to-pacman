def get_layout():
    # aa = '1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0 1 1 1 1 0 1 1 0 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 0 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1'
    # aa = '1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 1 1 0 1 0 0 0 0 1 0 1 1 0 1 0 0 0 0 1 0 1 1 0 1 0 0 0 1 0 0 1 1 0 1 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1'
    aa ='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + - - - - + - - - - - + 0 0 + - - - - - + - - - - + 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 + - - - - + - - + - - + - - + - - + - - + - - - - + 0 0 0 0 0 | 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 | 0 0 0 0 0 + - - - - + 0 0 + - - + 0 0 + - - + 0 0 + - - - - + 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 + - - + - - + - - + 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 + - - - - - + - - + 0 0 0 0 0 0 0 0 + - - + - - - - - + 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 + - - - - - - - - + 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 + - - - - + - - + - - + 0 0 + - - + - - + - - - - + 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 | 0 0 | 0 0 0 0 0 | 0 0 0 0 | 0 0 0 0 0 + - + 0 0 + - - + - - + - - + - - + - - + 0 0 + - + 0 0 0 0 0 0 0 | 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 | 0 0 0 0 0 0 0 0 | 0 0 | 0 0 | 0 0 0 0 0 0 0 + - + - - + 0 0 + - - + 0 0 + - - + 0 0 + - - + - + 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 0 | 0 0 | 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 + - - - - - - - - - - + - - + - - - - - - - - - - + 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    nn = 31
    xx = 0
    yy = 0
    lstlst = []
    for ii in aa:
        if ii == ' ':
            continue
        elif ii =='0':
            lstlst.append([+1*(((xx%nn)*(2/nn))-1)+0.1,-1*(((yy%nn)*(2/nn))-1)])
            xx+=1
            if xx >nn-1:
                yy+=1
                xx-=nn  
        else: 
            xx+=1
            if xx >nn-1:
                yy+=1
                xx-=nn
    return lstlst,len(lstlst)