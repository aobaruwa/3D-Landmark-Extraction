def read_lnd(fname):
    lnd_data=open(fname,'r').readlines()
    start=False
    landmark_dict={}
    for l in lnd_data:
        if l.startswith("AUX ="):
            start=True
            continue
        if l.startswith("END ="):
            break
        if start:            
            data=l.split()
            # No documentation on what these 3 numbers are, but d2 is -999 if the point is missing
            d2=data[2]
            x=float(data[4])/1000.
            y=float(data[5])/1000.
            z=float(data[6])/1000.
            if d2 == "-999": continue
            name=" ".join(data[7:])
            landmark_dict[name]=(x,y,z)
            
    labels = []
    f = open("landmark_names.txt", "r")
    names = [x.strip() for x in f.readlines()]
    for name in sorted(names):
        labels.extend(landmark_dict.get(name, (-999,-999,-999)))
    assert len(labels)==225
    f.close()

    return labels
